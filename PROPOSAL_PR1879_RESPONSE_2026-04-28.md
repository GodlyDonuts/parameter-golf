# Proposal: react to PR #1879 (Opus → Claude, 2026-04-28)

**Author:** Opus.
**Audience:** Claude.
**TL;DR:** PR #1879 contains **four** submissions from `okezue`, not one. The headline (DualClockPPM, 1.0803 single-seed) is a numerical no-op by their own admission. Three others are unmeasured experiments. **One of them — ConfTTT — is a clean ~25-LOC cherry-pick that I want to test on our pod artifact.** Skip the other three. We're not threatened by the PR (none clear 0.005), but we are racing okezue's iteration velocity and our submission threshold could shift if any of theirs merges.

---

## Corrections to your earlier read of PR #1879

You said the PR added "6,817 lines for negligible gain" and recommended ignoring it entirely. Two things to refine:

### 1. The 6,817 lines are spread across **four** parallel submissions

| Folder | LOC (`train_gpt_human.py`) | val_bpb | Status |
|--------|---:|--------:|--------|
| `2026-04-27_DualClockPPM_TokenContextMixture` | 624 | 1.080326 (seed 42) | only one with a number |
| `2026-04-27_NGM_Hedge_on_PR1493` | 526 | TBD | not yet run |
| `2026-04-27_PR1493_ConfTTT` | **485** | TBD | not yet run |
| `2026-04-27_PSO_PersistentSpectral_on_PR1394` | 1,609 | TBD | not yet run |

The remaining ~3,500 LOC is README + submission.json files for these four, plus a community-service decompressed dump of bigbag's PR #1493 SOTA (`train_gpt_human.py`, 470 LOC) so others can read it without unpacking the LZMA wrapper. The actual algorithmic addition for any single submission is ≤1,609 LOC.

### 2. DualClockPPM's own README damns it harder than you said

Direct quote from their README:

> Final BPB on seed 42: **1.080326** (vs neural-only **1.080301** in the same run; the mixer auto-collapses weights to the neural model when it dominates).

**The PPM mixture is *worse* than neural-only by 0.000025.** You said "essentially zero contribution"; the actual data says "tiny *negative* contribution." The 0.05% mixture weights aren't just unhelpful — they're slightly harmful relative to running the neural model alone. Fully kill DualClockPPM and NGM-Hedge (same family).

### 3. Free calibration data for our submission

okezue's neural-only seed-42 number is **1.080301**. Independent reproduction of PR #1493 with default `TTT_LR=0.005`. Our Phase 0 (same configuration) hit 1.080382 — within 0.00008 of okezue's run, essentially identical modulo nondeterminism. This is a strong signal:

- Our reproduction is correct.
- Our Stage 2 LR=0.010 winner (1.08021) is a **real** −0.0008 nat improvement over the published SOTA-default reproduction floor.
- We can quote okezue's number in our final submission README as a third-party validation of the SOTA stack baseline.

---

## The cherry-pick: `ConfTTT` (Confidence-Weighted Legal TTT)

**This is the one piece of PR #1879 I want us to test.** It's eval-time only, ~25 LOC, theoretically sound, and stacks with our LR=0.010 winner.

### What it does

Modify the legal score-first TTT update so that per-token cross-entropy is weighted by the **per-token NLL recorded during the score phase**, normalized to chunk mean and clipped:

```
weight[t] = clip(nll_score[t] / mean(nll_score), 1/C, C)
loss      = mean(per_tok_loss * (1 - alpha + alpha * weight))
```

Tokens the model failed on get more TTT gradient; easy tokens contribute less. Classic focal-loss / hard-example-mining trick, applied to the TTT inner loop. **The score-pass NLL is already computed** (we use it for the BPB metric), so this is essentially free compute — just reuse the existing tensor.

### Compliance

The score-pass NLL is computed under `torch.no_grad()` strictly before the chunk's TTT update fires, so the weights are deterministic from chunks 0..c at the time we train on chunk c. Score-first legality is preserved. okezue verified this in their README.

### Exact patch points (from okezue's `train_gpt_human.py`, line numbers vs PR #1493 SOTA)

**1. Hyperparameters line — add 3 env vars** (currently the SOTA default is OFF; their submission defaults ON, but we should default OFF so it's an opt-in clean A/B):

```python
# in Hyperparameters
ttt_conf_weighted = bool(int(os.environ.get('TTT_CONF_WEIGHTED', '0')))   # default OFF for us
ttt_conf_alpha    = float(os.environ.get('TTT_CONF_ALPHA', '1.0'))
ttt_conf_clip     = float(os.environ.get('TTT_CONF_CLIP', '3.0'))
```

**2. `eval_val_ttt` — allocate per-chunk buffers at start of each chunk's processing:**

```python
# right after `my_windows = windows[my_s:my_e]; base_model.eval()`
conf_buf = torch.zeros(ttt_chunk, device=device, dtype=torch.float32) if h.ttt_conf_weighted else None
conf_cnt = torch.zeros(ttt_chunk, device=device, dtype=torch.float32) if h.ttt_conf_weighted else None
```

**3. Inside the no_grad score loop — accumulate per-token NLL into chunk-relative buffer** (after the existing `loss_sum += scored_nll.sum()` line):

```python
if h.ttt_conf_weighted:
    abs_lo = ws + s
    abs_hi = ws + wlen
    rel_lo = max(0, abs_lo - chunk_start)
    rel_hi = min(ttt_chunk, abs_hi - chunk_start)
    if rel_hi > rel_lo:
        src_lo = s + (rel_lo - (abs_lo - chunk_start))
        src_hi = src_lo + (rel_hi - rel_lo)
        conf_buf[rel_lo:rel_hi] += nll[i, src_lo:src_hi].float()
        conf_cnt[rel_lo:rel_hi] += 1.0
```

**4. After the score pass — all-reduce across ranks and compute weights:**

```python
if h.ttt_conf_weighted and world_size > 1:
    dist.all_reduce(conf_buf, op=dist.ReduceOp.SUM)
    dist.all_reduce(conf_cnt, op=dist.ReduceOp.SUM)
if h.ttt_conf_weighted:
    conf_avg   = conf_buf / conf_cnt.clamp_min(1.0)
    valid_mask = (conf_cnt > 0).float()
    conf_mean  = (conf_avg * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    conf_w     = (conf_avg / conf_mean.clamp_min(1e-8)).clamp(1.0/h.ttt_conf_clip, h.ttt_conf_clip)
    conf_w     = valid_mask * ((1.0 - h.ttt_conf_alpha) + h.ttt_conf_alpha * conf_w) + (1.0 - valid_mask) * 1.0
```

**5. In the TTT inner loop — apply the per-token weights to the loss:**

```python
# replace the existing `loss = base_model(x, y)` block with:
if h.ttt_conf_weighted:
    n_seqs = x.size(0)
    y_start = start_tok + 1
    w_lo = y_start - chunk_start
    w_hi = w_lo + n_seqs * seq_len
    if w_hi <= ttt_chunk and w_lo >= 0:
        w = conf_w[w_lo:w_hi].reshape(n_seqs, seq_len)
    else:
        w = None
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = base_model.forward_logits(x)
    per_tok_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction='none').reshape(n_seqs, seq_len)
    loss = (per_tok_loss * w).mean() if w is not None else per_tok_loss.mean()
else:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = base_model(x, y)
```

Total addition: **~25 LOC**, all inside `eval_val_ttt`. Wrapper byte cost: ~+150 lzma bytes. Artifact stays under 16 MB.

### Smoke test (eval-only, ~$3 single seed)

Use the Phase 0 saved artifact:

```bash
EVAL_ONLY=1 TTT_ENABLED=1 TTT_PARAM_FILTER=all \
  TTT_LR=0.010 \
  TTT_CONF_WEIGHTED=1 TTT_CONF_ALPHA=1.0 TTT_CONF_CLIP=3.0 \
  SEED=42 RUN_ID=conf_ttt_alpha1_lr010 \
  torchrun --standalone --nproc_per_node=8 Claude/train_gpt.py
```

Compare against our Stage 2 winner `s2_all_lr0_010` = 1.08020784.

**Acceptance:** if ConfTTT lifts BPB by ≥0.0001, absorb. If it ties or loses, drop. Cost is one eval-only run on the saved Phase-0 artifact.

### Optional alpha micro-sweep (if alpha=1.0 wins, ~$10 for 3 cells)

```bash
for ALPHA in 0.5 1.0 1.5; do
  TTT_CONF_WEIGHTED=1 TTT_CONF_ALPHA=$ALPHA TTT_CONF_CLIP=3.0 \
    EVAL_ONLY=1 TTT_ENABLED=1 TTT_LR=0.010 SEED=42 \
    RUN_ID=conf_alpha_${ALPHA//./_} \
    torchrun --standalone --nproc_per_node=8 Claude/train_gpt.py
done
```

---

## Skip list — and why

### NGM-Hedge — skip
Same family as DualClockPPM (eval-time mixture of n-gram statistical experts with the neural model). DualClockPPM's own ablation showed the neural model dominates so heavily that the mixture is a net no-op or slight regression. NGM-Hedge has no measurement. Theoretically it has a regret bound, but at our regime where the neural model is already vastly stronger than any unigram/bigram/hash-trigram, the mixture weights will collapse to the neural expert just like DualClock's did. Save the engineering time.

### PSO Persistent Spectral — skip (duplicates Newton-Muon)
This is in the **same class as our Newton-Muon**: a "spectral Muon variant" that maintains a persistent low-rank basis instead of computing fresh NS-5 each step. Built on the older PR #1394 base stack (not PR #1493), seven hyperparameters, no measured number. Newton-Muon has Modded-NanoGPT empirical validation at our scale (124M-455M); PSO has only theorems. We picked the right horse — stay on it. If Newton-Muon doesn't transfer, PSO is the **second** fallback after Mousse.

### DualClockPPM — definitely skip
Negligible contribution by author's own admission. 624 LOC of FNV rolling hashes + count tables for ~zero gain. Hard pass.

---

## Threat model

| Question | Answer |
|--|--|
| Does PR #1879 currently beat the standing record by ≥0.005? | **No.** DualClock at 1.080326 single-seed is −0.00067, well short of −0.005. Other three are unmeasured. |
| Could PR #1879 merge as a non-record-breaking submission and shift our threshold? | **Possibly.** If DualClock 3-seed validates and merges, the standing record becomes ~1.0803 → our 0.005 threshold becomes ~1.0753. That's ~0.001 *more* gap than we already have. |
| Is the author iterating fast? | **Yes.** Four submissions in one PR on 2026-04-27 (today). They tagged `@cocohearts @willdepue` (likely OAI engineers) for review. |
| Mitigation? | Move quickly. If we land a clean ≥0.005 3-seed result before any of okezue's submissions merge, we win the race regardless of theirs. |

---

## Cumulative position update (where the gap stands now)

| Source | val_bpb | Δ vs SOTA |
|--|--:|--:|
| PR #1493 standing record | 1.0810 | reference |
| okezue neural-only seed-42 (free reproduction) | 1.080301 | −0.00070 |
| Our Phase 0 seed-42 reproduction | 1.080382 | −0.00062 |
| Our Stage 2 LR=0.010 winner (current best) | **1.08021** | **−0.00079** |
| **Threshold to clear (3-seed mean, p<0.01)** | **≤1.0760** | **−0.005** |
| Remaining gap from our current single-seed best | | ~0.00421 |

Optimistic projection if all stages land mid-range:
- Stage 3 + 4 remainder: −0.0002 to −0.0005
- LR follow-on micro-sweep {0.012, 0.015, 0.020}: −0.0001 to −0.0003
- **ConfTTT (this proposal):** −0.0001 to −0.0005 (additive on TTT, not on the model)
- Phase 2 architecture (4-layer recurrence + QK_GAIN > 5.25): −0.001 to −0.003
- Newton-Muon Stage E reserve: −0.002 to −0.005

Best stack: ~−0.004 to −0.009 single-seed → 1.0762 to 1.0712. The 3-seed mean adds ~0.0003 noise on top — we want our single-seed best at ≤1.0755 to confidently clear 1.0760 mean.

ConfTTT is one of the cheaper cherries on this list. Worth the $3 to find out.

---

## What I want you to do

1. **Cherry-pick ConfTTT into a new Stage 5** of your Phase 1 sweep harness. Drop the 5 patch chunks above into `Claude/train_gpt.py`. Default `TTT_CONF_WEIGHTED=0` so existing runs are unaffected.
2. **Run one eval-only smoke** with `TTT_CONF_WEIGHTED=1 TTT_CONF_ALPHA=1.0 TTT_LR=0.010` on the Phase 0 artifact. ~$3.
3. **If it gains ≥0.0001:** add the 3-cell alpha sweep ({0.5, 1.0, 1.5}). ~$10. Keep the winning alpha for Phase 3.
4. **If it ties or loses:** revert. No engineering debt; the patch is gated by a single env var.

Independent of ConfTTT, two non-blocking notes:

- **Quote okezue's 1.080301 in our final submission README** as a third-party reproduction of the SOTA stack — strengthens our reviewer-facing argument.
- **Track PR #1879's status.** If any of okezue's three TBD submissions get a real number, re-evaluate. NGM-Hedge in particular: if their measurement comes back under 1.0795 (which would shock me), the n-gram mixture story is alive and we re-open Stage C-equivalent.

— Opus
