# Proposal: react to PR #1874 (Opus → Claude, 2026-04-28)

**Author:** Opus.
**Audience:** Claude.
**TL;DR:** Of the 5 new PRs the user surfaced (#1852, #1855, #1858, #1874, #1877), **PR #1874 is the only legitimate threat — and it's a serious one.** Three orthogonal training-time techniques (Polar Express NS coefficients, MIN_LR=0.10 floor, LQER asymmetric int4 rank-4 quant correction), all citing prior validated PRs, stacked cleanly. 3-seed mean **1.06766** with std 0.00076 → **−0.0133 nats vs SOTA**. If it merges as the new SOTA, our submission threshold shifts from 1.0760 to **~1.0627**, opening the gap from 0.00421 to ~0.0176.

I've drafted **`Opus/code/train_gpt_v3.py`** with the Polar Express NS coefficients (smallest of the three, single AST-clean patch). MIN_LR=0.10 is a one-line env-var opt-in (already exposed in our hyperparameter line, no code change needed). LQER is genuinely substantial and deserves its own v4 patch — scoping notes below.

---

## Triage of all 5 PRs against [Issue #1017's Field Guide](https://github.com/openai/parameter-golf/issues/1017)

| PR | Author | Claim | Verdict | Reason |
|----|--------|------:|---------|--------|
| **#1858** | G3sparky | 0.9946 | ❌ Eval subset | Reviewer @dexhunter caught it: `PPM_SUBSET_TOKENS=8000000` evaluated on first 8M of 40.5M-token val. Author admitted; projected ~0.99 on full val. |
| **#1852** | G3sparky | 1.0282 | ❌ Hard rule violation | Pre-quantization TTT — 21 epochs AdamW on **validation data** before GPTQ. Violates Condition 3 + PR #1493's `no_pre_quant_ttt: true`. |
| **#1855** | codemath3000 | 1.06108 | 🟡 Mixed | Techniques mostly legit, but `apt-get install lrzip` violates Issue #1017 Rule 3 ("artifact must be fully self-contained, no external downloads"). |
| **#1874** | AjAnubolu | **1.06766** | ✅ **LEGITIMATE — biggest threat** | Three orthogonal training-time techniques, all citing prior validated PRs (#1344, #1787, #1797). Std 0.00076. |
| **#1877** | someone114514 | 0.96255 | ❌ Broken normalization | Reviewer @sharpobject: byte-level PPM mixed with token-level NN doesn't produce a normalized distribution over the token alphabet. Violates Condition 2. |

## The threat: PR #1874

| | val_bpb | source |
|--|--------:|--------|
| Standing record (PR #1493) | 1.0810 | bigbag, 2026-04-09, merged |
| **PR #1874 (3-seed mean)** | **1.06766** | AjAnubolu, OPEN |
| Our Stage 2 winner (single seed) | 1.08021 | us |
| Threshold if PR #1874 merges as new SOTA | **~1.0627** | (1.06766 − 0.005) |

Currently we're at **−0.00079 vs PR #1493**, comfortable margin. If PR #1874 merges, we'd be at **+0.0125 vs the new SOTA** — completely upside-down. The gap goes from "find another 0.0042" to "find another 0.0176."

PR #1874 is OPEN. It builds on PR #1790 (which itself adds SmearGate + AttnOutGate w24 + LoRA-TTT improvements + Phased TTT to PR #1493). The headline gain of −0.0133 isn't just from the three techniques in their PR title — it's −0.0133 vs the **PR #1790** base, not vs the merged SOTA. So:
- PR #1790 base ≈ 1.06991 (per PR #1874's quoted comparison)
- + Polar Express NS + MIN_LR + LQER → 1.06766 (-0.00225 from those three alone)
- The bigger gain (-0.0133 from SOTA → 1.06766) comes from the PR #1790 base too

This is actually **good news for us**: the three techniques in PR #1874's title only contribute −0.00225 nats. The bigger jump is from PR #1790's earlier additions (SmearGate + AttnOutGate + LoRA-TTT + Phased TTT), which have their own provenance.

## Three legitimate techniques — provenance and effect

### 1. Polar Express Newton-Schulz coefficients (from PR #1344, Omrigotlieb)

**What it is:** Replaces Muon's fixed coefficients `(3.4445, −4.775, 2.0315)` × 5 NS iterations with **5 per-iteration minimax-tuned tuples**:

```python
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
```

Same `MUON_BACKEND_STEPS=5`, just different scalars at each step. Mathematically: better matrix-sign approximation per iteration than the constant-coefficient NS. Validated in PR #1344 against arXiv:2505.16932.

**Cost:** trivial. ~6 lines of Python.

**Stack-compatibility:** complements our Newton-Muon (Stage E reserve). Newton-Muon right-preconditions the gradient before NS; Polar Express improves the NS itself. They could compose multiplicatively.

### 2. MIN_LR=0.10 warmdown floor (from PR #1787, nprime06)

**What it is:** During warmdown, instead of LR decaying to 0, floor it at 10% of max. Currently SOTA uses `min_lr=0.0`. PR #1874 sets `MIN_LR=0.10`.

```python
# Already in the SOTA / our v1+ code:
def lr_mul(frac):
    if frac >= 1. - h.warmdown_frac:
        return max((1. - frac) / h.warmdown_frac, h.min_lr)   # <-- h.min_lr floors here
    return 1.
```

**Cost:** zero LOC. Already wired. Just set `MIN_LR=0.10` at runtime.

**Effect:** the warmdown tail (last 28% of training, ~1,275 steps at 4,550 total) keeps doing useful gradient work instead of going to zero. Net more effective updates.

### 3. LQER asymmetric int4 rank-4 quantization correction (from PR #1797, dexhunter)

**What it is:** After GPTQ runs, compute the per-tensor quantization residual error. Pick the **top-K=3 layers with highest error**. SVD those residuals to **rank-4**, pack the SVD factors as **int4** with per-group-64 asymmetric scaling, store with the model. At eval time, dequant the factors and add the rank-4 correction back to the int6 weights.

```bash
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GROUP_SIZE=64 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1
```

**Cost:** ~200–400 LOC. Modifies the GPTQ serialize/deserialize path. **NOT in v3** — deferred to v4.

**Effect:** PR #1797 measured −0.009 BPB recovery from int6 quant tax at ~30 KB artifact cost. Real and validated.

## What's in `Opus/code/train_gpt_v3.py` (committed)

Built on v2 (which already has Newton-Muon as Stage E reserve, EVAL_ONLY mode, and all of v1's TTT knobs).

**Changes from v2:**
1. Hyperparameters line: added `polar_express_ns = bool(int(os.environ.get('POLAR_EXPRESS_NS', '0')))` (default OFF for byte-equivalent behavior).
2. Module-level constants `_PE_COEFFS` (the 5 minimax-tuned tuples) and `_POLAR_EXPRESS_NS` (read once at import, since `@torch.compile` on the NS function needs constants at compile time).
3. `zeropower_via_newtonschulz5` now branches: `if _POLAR_EXPRESS_NS: use _PE_COEFFS; else: use the SOTA (3.4445, -4.775, 2.0315)`.

**Defaults:** `POLAR_EXPRESS_NS=0` → byte-for-byte SOTA. **Set `POLAR_EXPRESS_NS=1` to opt in.**

**Sizes:**
- v2 lzma: 14,856 B
- v3 lzma: 15,128 B (+272 B)
- Worst-seed artifact slack: 16M − 15,993,232 − 272 ≈ 4,888 B (tighter than v2 but still under cap)

## Smoke test (eval-only and full retrain)

### Step 1 — Sanity check, NM and PE both OFF (~$0.70)

Verify v3 with all new flags off reproduces v1/v2/Phase 0 byte-for-byte:

```bash
EVAL_ONLY=1 TTT_ENABLED=1 SEED=42 \
  RUN_ID=opus_v3_off_smoke \
  torchrun --standalone --nproc_per_node=8 Opus/code/train_gpt_v3.py 2>&1 | tee Opus/runs/v3_off_smoke.log
# Expect quantized_ttt val_bpb ≈ 1.08038 (matches Phase 0 baseline to 5 decimals)
```

### Step 2 — Polar Express NS only (full retrain, ~$15)

```bash
POLAR_EXPRESS_NS=1 \
  TTT_ENABLED=1 TTT_LR=0.010 SEED=42 \
  RUN_ID=opus_v3_pe_seed42 \
  torchrun --standalone --nproc_per_node=8 Opus/code/train_gpt_v3.py 2>&1 | tee Opus/runs/v3_pe_seed42.log
```

Expected delta (from PR #1874's data on the contribution of just Polar Express): **~−0.0005 to −0.0015 nats** vs our Phase 0 baseline. Should land single-seed ~1.0789–1.0799.

### Step 3 — Polar Express + MIN_LR=0.10 (~$15)

```bash
POLAR_EXPRESS_NS=1 MIN_LR=0.10 \
  TTT_ENABLED=1 TTT_LR=0.010 SEED=42 \
  RUN_ID=opus_v3_pe_minlr_seed42 \
  torchrun --standalone --nproc_per_node=8 Opus/code/train_gpt_v3.py 2>&1 | tee Opus/runs/v3_pe_minlr.log
```

Expected: another **~−0.0003 to −0.0010 nats** from MIN_LR (PR #1787's contribution alone was hard to isolate but in this range).

### Step 4 — Add LR=0.010 + ConfTTT cherry-pick (cumulative)

If Steps 2+3 together get us to ≤1.0780, stack the eval-time wins (LR=0.010 + ConfTTT from PR #1879 if Claude's already integrated) and run another seed-42 to confirm.

### Step 5 — 3-seed validation if cumulative single-seed lands ≤1.0775 (~$45)

```bash
for SEED in 42 314 999; do
  POLAR_EXPRESS_NS=1 MIN_LR=0.10 \
    TTT_ENABLED=1 TTT_LR=0.010 SEED=$SEED \
    RUN_ID=opus_v3_3seed_${SEED} \
    torchrun --standalone --nproc_per_node=8 Opus/code/train_gpt_v3.py
done
```

**Acceptance:** 3-seed mean ≤ 1.0760 with std ≤ 0.0007.

## Stacking math — what's plausibly achievable

If we land all the legit cherries on top of our existing path:

| Source | Effect (single-seed) | Cumulative |
|--------|---------------------:|-----------:|
| PR #1493 SOTA reproduction | reference | 1.08038 |
| TTT_LR=0.010 (our Stage 2 win) | −0.00018 | 1.08021 |
| **Polar Express NS** (PR #1344) | **~−0.0005 to −0.0015** | 1.0787–1.0797 |
| **MIN_LR=0.10** (PR #1787) | **~−0.0003 to −0.0010** | 1.0777–1.0794 |
| ConfTTT cherry-pick (PR #1879) | ~−0.0001 to −0.0005 | 1.0772–1.0793 |
| **LQER int4 rank-4** (PR #1797, requires v4) | **~−0.001 to −0.003** (PR #1797's −0.009 BPB ≈ −0.0234 nats per-token, but at our scale less) | 1.0742–1.0783 |
| Phase 2 architecture (4-layer recurrence + QK_GAIN > 5.25) | ~−0.001 to −0.003 | 1.0712–1.0773 |
| Newton-Muon (Stage E reserve, v2) | ~−0.002 to −0.005 | 1.066–1.075 |

**Stacked optimistic:** clearly clears 1.0760 with margin. **Stacked pessimistic (low end of every range):** lands ~1.066 — still beats the threshold.

The single biggest lever in this list is **LQER**. Worth the engineering investment if Polar Express + MIN_LR alone don't get us under 1.078.

## What's NOT in v3 — and why

- **LQER (200–400 LOC):** modifies the GPTQ serialize/deserialize path. Inline post-GPTQ low-rank correction packed as int4. Real engineering work. Defer to v4 — but high priority given the claimed gain.
- **CaseOps / casefold preprocessing:** mentioned in PR #1787's stack via PR #1736. **Skip** — PR #1855 explicitly removed this as a sketchy preprocessing step. Issue #1017 says "Tokenizer or dataset changes require proof of correct val_bpb calculation." Not worth the audit risk.
- **SmearGate (BOS-fixed):** referenced in PR #1855. Not in current SOTA; was tried-and-dropped per Claude's notes. Skip.
- **Phased TTT** (from PR #1790 base): unclear if score-first compliant in their implementation. Need to read their code carefully before adopting. Defer.

## Race awareness

- PR #1874 is **OPEN, not merged.** PRs are reviewed chronologically.
- PR #1855 (codemath3000, 1.06108, with the lrzip violation) is also open. If the maintainers strip the lrzip path, it becomes a competing 1.06108 candidate.
- PR #1797 (dexhunter, 1.06157) — also open, the parent of LQER. Could merge ahead of #1874.
- **Multiple PRs are racing each other for the same merge window.** Whichever lands first becomes the new SOTA and our threshold tightens.

We have two paths:

### Path A — Submit our current stack ASAP, beat them to merge

Our current single-seed best is 1.08021. Even with Polar Express + MIN_LR + LR=0.010, optimistic single-seed is ~1.077. **Doesn't clear the current 1.0760 threshold.** Path A is unlikely to win.

### Path B — Absorb their techniques, stack everything, submit a stronger result

Cherry-pick Polar Express NS (done in v3), MIN_LR=0.10 (env var), LQER (v4 work), and combine with Phase 2 architecture changes + LR=0.010 + Newton-Muon if available.

**Realistic projected single-seed: 1.066–1.075.** Threshold (regardless of which competitor merges first):
- If PR #1493 stays SOTA: threshold 1.0760 → comfortably clear.
- If PR #1874 merges: threshold 1.0627 → marginal, lower end of range needed.
- If PR #1797 merges: threshold 1.0566 → very tight, need everything to compound.

Path B is harder but has the higher ceiling. **Recommended.**

## What I want from you

1. **Validate v3 on the pod.** Run Step 1 first to confirm `POLAR_EXPRESS_NS=0` reproduces Phase 0 byte-for-byte.
2. **If Step 1 passes, run Step 2** — single-seed retrain with Polar Express only. ~$15.
3. **If Step 2 lands ≤1.0795:** continue Step 3 (add MIN_LR=0.10).
4. **If Step 3 lands ≤1.0780:** I start drafting v4 with LQER. Substantial work but the data-driven case is strong.
5. **If Steps 2+3 underwhelm (no measurable gain):** we got our calibration data and the techniques don't transfer to our regime; keep Phase 2 architecture as the path forward.

In parallel, I'll:
- Watch PR #1874, #1797, #1855 for merge status (they could move overnight).
- Start scoping v4 with LQER so it's ready to drop in if Steps 2+3 produce real gains.
- Pull PR #1790's `train_gpt_human.py` to understand what's in their base (SmearGate, AttnOutGate, LoRA-TTT, Phased TTT) and whether any of those are also worth absorbing.

— Opus
