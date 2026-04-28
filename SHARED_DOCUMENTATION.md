# SHARED_DOCUMENTATION.md — agent crib sheet

A living document any AI working on this challenge can read. Add findings, bugs, and infra gotchas in chronological order. Don't rewrite history — append.

**Challenge:** OpenAI Parameter Golf (track_10min_16mb).
**Current SOTA to beat:** `val_bpb = 1.0810` (PR #1493, bigbag, 2026-04-09).
**Threshold:** ≥0.005 nats with p<0.01, 3-seed mean.
**Deadline:** 2026-04-30.

## Folder ownership

| Folder | Owner | Contents |
|--------|-------|----------|
| `Claude/` | Claude (me) | Patched `train_gpt.py`, sweep scripts, runpod helper, run logs |
| `Opus/` | Opus | Plan + experiment specs + selective-TTT impl in `code/train_gpt_v1.py` |
| `Antigravity/` | (older agent) | MLA + 3.5× MLP + Int6 QAT — non-record-track stack |
| `records/` | OpenAI repo | Read-only reference; do not modify |
| `data/` | OpenAI repo | Tokenizer + cached fineweb downloader |
| **root** | shared | This file |

## Agreed-on conventions (so we don't trip over each other)

- The **base SOTA** stack lives at `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` — that's the LZMA-wrapped one-line submission. The decompressed source is 469 lines, 48,583 B raw → ~16,650 B compressed.
- Anyone editing `train_gpt.py` should keep a **decompressed copy** in their folder (e.g. `Claude/sota_train_gpt.py`) for diff/reference. Don't mutate the records folder.
- All record submissions are at `train_seq_len=2048`, `vocab_size=8192`, 11 layers × 512d × 8H/4KV-head, MLP×4. Don't break these unless you mean to.
- LZMA wrapper budget: total artifact ≤ **16,000,000 decimal bytes**. SOTA's 3 seeds were 15,991,930–15,993,232 → ~7 KB headroom on the model side.

## RunPod gotchas

### The pod template matters
The official **Parameter Golf RunPod template** ID is `y5cejece4j` (referenced in repo README). It ships with:
- Python 3.12.3
- torch 2.9.1+cu128
- flash_attn_3 (`flash_attn_interface`) importable
- 100 GB container disk
- /workspace volume disk

It does **NOT** ship with `brotli` or `zstandard` (compressors used in `_compress`). You need:
```bash
pip install --break-system-packages brotli zstandard
```
PEP 668 will block plain `pip install` — `--break-system-packages` is required because the system Python is externally managed.

### Container disk size
Default is **10 GB which is too small** — pip cache + HF cache default to `/root/.cache/*` on the container disk. With 10 GB, both data download (6 GB transient) and torch install (3 GB+) hit `No space left on device`.
**Recommended pod spec:** ≥50 GB container disk + 30 GB /workspace volume. We've been running on 100 GB / 30 GB.

### SSH key auto-injection
RunPod injects your registered public keys into a pod's `~/.ssh/authorized_keys` **only at pod creation time**. Adding a key to the account *after* a pod exists won't propagate.
Workaround: paste the public key directly into `~/.ssh/authorized_keys` via the web shell.

### SSH session 4-min cap
RunPod can drop SSH sessions after ~4 minutes. Long jobs **must** be detached:
```bash
nohup bash <script> > <log> 2>&1 &
```
We have a `launch_nohup.sh` and a `runpod.sh` helper that do this.

## Code quirks of the SOTA train_gpt.py

- **Python 3.12-only syntax:** the SOTA uses nested-quote f-strings at lines 266 and 456 (e.g. `f"{', '.join(...)}"` with both inside double quotes). On Py 3.11, these fail with `SyntaxError: f-string: expecting '}'`. Fix: change inner strings to single quotes (cleanly equivalent).
- **`grad_accum_steps = 8 // world_size`** — so on `nproc_per_node=1` you get `grad_accum=8`, *different* training dynamics from 8×H100. Don't try to reproduce SOTA on 1×H100; the baseline number won't match.
- **Tied embeddings init std = 0.005**, not the usual 0.02. Custom for this stack.
- **`CONTROL_TENSOR_NAME_PATTERNS`** at line 199 is the existing partition for routing fp32 control tensors away from Muon. It's also exactly the right surface for selective-TTT (see Opus's `_select_ttt_params`).
- **TTT mutates the dequantized model**, not the int6 grid. So `eval_val_ttt` is fp32 SGD on the post-deserialize model. Don't reason as if quantization is being re-applied.
- **`enable_looping_at` defaults to 0.35** of training fraction; before that the model uses non-looped forward. There's a separate `loop_warmup` pass (in main()) that warms up the looped graph once before the main run. Both warmups run for `WARMUP_STEPS=20` each.

## Active patches in `Claude/train_gpt.py` (vs the pristine SOTA)

1. **`EVAL_ONLY=1`** guard in `train_and_eval` — skips training, reuses an existing `final_model.int6.ptz`. For cheap TTT eval-only sweeps.
2. **`TTT_PARAM_FILTER`** ∈ {all, scales, scales+embed, last_n_layers:K, attn_only, mlp_only}. `scales` reuses `CONTROL_TENSOR_NAME_PATTERNS`. Adopted from Opus's design.
3. **`TTT_WD`** — adds weight decay to the SGD optimizer used in TTT.
4. **`TTT_RESET_MOM`** — zeroes the SGD momentum buffer at chunk boundaries.
5. **Single-quote fixes** at lines 266 + 456 for Py3.11 compatibility (no behavioral change — strings produce identical output).

Wrapper size: 17,115 B (+508 vs SOTA). Still within 16 MB cap by ~7 KB.

## What's been tried on the leaderboard, what to skip

(Distilled from `Claude/notes/leaderboard_intel.md`.)

**Already explored & dropped** (don't waste compute):
- Ternary / 1-bit quantization
- FP16 embeddings (int8 GPTQ wins)
- BigramHash (SP8192 makes it redundant)
- YaRN, NeoMuon (incompatible with Muon stack)
- SmearGate (negative interaction with vocab+recurrence)
- Value embeddings VE128
- Coprime-stride loader
- ±1 selective pruning
- Late QAT, GPTQ-lite (full Hessian GPTQ wins)
- MTP (multi-token prediction)
- Long seq_len (4096+) eval
- Hash embeddings
- Param banking / Parallel Muon

**Untouched at the current SOTA stack — open territory:**
| Idea | Cost | Expected EV |
|------|------|-------------|
| Selective-TTT (filter ∈ scales/...) | $20 eval-only | medium-high |
| 4-layer recurrence (LOOP_END=6) | $5 (1 retrain) | medium |
| 3-layer × 3 loops | $5 | medium |
| QK_GAIN_INIT > 5.25 | $7 | low-med |
| TTT chunk size other than 32K | included in sweep | low-med |
| TTT with weight_decay | included in sweep | low |
| TTT with momentum reset | included in sweep | low |
| SWA over last 10% | new code | medium (untried) |

## TTT sensitivity warning

PR #756 documents **25 failed TTT attempts** before Score-First TTT worked. TTT placement in the stack matters — naive full-param SGD on an arbitrary checkpoint can hurt. The 1.0810 SOTA uses a specific Score-First ordering (score chunk N under no_grad, *then* update on chunk N's tokens). Any TTT variant should preserve this ordering.

## Live run state (Phase 0 / 1 / 2 / 3)

### 2026-04-27 23:27 UTC — Phase 0 first attempt
Killed early — `run_phase0_baseline.sh` was running the **wrong train_gpt.py** (the stock 1.22 BPB baseline at repo root, not `Claude/train_gpt.py`). Bug was in the runner script's `torchrun ... train_gpt.py` line. **Fix:** `torchrun ... Claude/train_gpt.py`. All 4 phase scripts patched.

### 2026-04-27 23:29 UTC — Phase 0 second attempt
**Training succeeded:** pre-quantization post-EMA val_bpb = **1.08734537**. That's healthy (SOTA pre-quant is around the same). 4586 train steps in 588s, 8M tok/s, 35,944,536 model params.
**Failed at compression:** `ModuleNotFoundError: No module named 'brotli'`. The pod template ships torch+FA3 but not brotli. **Fix:** `pip install --break-system-packages brotli zstandard`.

### 2026-04-27 23:43 UTC — Phase 0 third attempt
Restarted clean after brotli install.
PIDs at launch: Phase 0 = 49993, auto_chain = 49994.
ETA: ~10 min train + ~8 min eval (sliding + TTT) = ~18 min total = ~$6.
Auto-chain validates `quantized_ttt val_bpb ∈ [1.0790, 1.0830]` and auto-fires Phase 1 if good.

## Phase plan (Claude's track)

| Phase | What | Cost |
|-------|------|------|
| 0 | reproduce 1.0810, save artifact | $5 |
| 1 | TTT eval-only sweep, 4 stages, ~18 configs | $40 |
| 2 | retrain with 4-layer recurrence + QK_GAIN sweep, 5 retrains | $17 |
| 3 | 3-seed final on Phase-2 winner | $15 |
| reserve | stretch ideas (SWA, mixed-bit GPTQ, etc.) | $400 |

See `Claude/STRATEGY.md` for full reasoning.

## Useful one-liners

```bash
# from project root, on local machine:
bash Claude/scripts/runpod.sh status              # ssh, gpu, procs, disk
bash Claude/scripts/runpod.sh tail phase0         # live log tail
bash Claude/scripts/runpod.sh push                # tar Claude/ → pod
bash Claude/scripts/runpod.sh pull                # tar Claude/runs ← pod
bash Claude/scripts/runpod.sh kill                # nuke all torchrun procs
bash Claude/scripts/runpod.sh phase0|phase1|phase2|phase3   # detached launch
```

## Append your findings below

---

### 2026-04-27 — Gemini Deep Research literature pass (Opus)

Full transcript at [DEEP_RESEARCH_2026-04-27.md](DEEP_RESEARCH_2026-04-27.md). Run by sairamen via Gemini Deep Research; my synthesis below. **Citations dated 2025–2026 are not independently verified — fetch the arXiv ID before committing GPU time.**

#### Top-3 actionable finds (highest EV / lowest cost in the 3-day window)

| Rank | Technique | Δ BPB (claim) | Surface | Lift |
|------|-----------|---------------|---------|------|
| 1 | **In-Place TTT** (NTP-aligned, MLP `down_proj` only, chunk-parallel outer-product update) | −0.004 to −0.008 | eval-time | new TTT *objective* (NTP outer-product, not CE) and *surface* (MLP final projection) |
| 2 | **qTTT** (cache K/V once, gradient only on Q projections) | −0.003 to −0.010 | eval-time | new TTT surface (Q projections only) — different from In-Place's MLP target |
| 3 | **Progressive Depth Recurrence** (stagger the looping activation, deep loops first) | −0.001 to −0.002 | train-time | one-line scheduling change, removes the loss spike at frac=0.35 |

**Headline:** Gemini ranks **In-Place TTT (arXiv:2604.06169, Feng et al. April 2026)** as the most novel finding the leaderboard hasn't absorbed. It's not the same as Opus's `scales` filter — In-Place TTT updates **MLP `down_proj`** with an **NTP outer-product target** (`W += η · V̂^T · Z` where `V_LM = E[x_{t+1}]`), not SGD on cross-entropy. Chunk-parallel, doesn't backward through quantized matrices, claims significant gain.

Implication for our work: the existing `TTT_PARAM_FILTER ∈ {all, scales, scales+embed, attn_only, mlp_only, last_n_layers:K}` covers *which params* but not *which objective*. In-Place TTT requires a separate code path for the NTP-aligned outer-product update.

#### Day-3 stretch (training-time, expensive — only if eval-time TTT runs come up short of −0.005)

| Technique | Δ BPB (claim) | Risk |
|-----------|---------------|------|
| **Mousse** (Shampoo-style preconditioner before Newton-Schulz orthogonalization, Muon drop-in) | −0.006 to −0.008 | 3% per-step overhead; needs hyperparam re-tune from SOTA's Muon settings |
| **NuMuon** (nuclear-norm budget on Muon updates → low-rank weights → better Brotli) | −0.002 to −0.006 | over-truncation hurts representational floor |
| **BHyT** norm (bounded tanh; 15.8% faster than RMSNorm) | −0.004 | activation distribution shifts → GPTQ recalibration needed |

#### Drop / don't bother (per Gemini)

- **AdEMAMix** — wins only at small-batch / noisy regime; loses to Muon/Mousse at our scale.
- **TTT-Linear / TTT-MLP (Stanford)** — requires replacing FA3. Architectural rewrite.
- **Attention sinks / register tokens** — designed for 100K+ context; pointless at seq_len=2048.
- **xIELU activation** — no fast kernels; Python autograd fallback blows 600s budget.
- **Test-time LoRA (post-quant)** — fp matrices at eval = memory + slow.

#### Verification needed before any GPU spend on these

The most aggressive arXiv IDs (2604.xxxxx, 2603.xxxxx, 2601.xxxxx) are claimed to be January–April 2026 papers. They may be real, may be hallucinated, or may be misattributed. **Anyone implementing one of these should:**

1. `WebFetch` arxiv.org/abs/<id> to confirm existence.
2. Cross-check the claimed mechanism + experimental scale (the regime needs to be ≤100M params, ≤350M training tokens, int6 quant — *not* a 7B-param finetune that doesn't transfer).
3. Treat the Δ BPB numbers as upper-bound priors — Gemini's ranges are wide.

#### What Opus is doing with this

I'm leaving Claude's Phase 0 / 1 / 2 / 3 plan untouched (Claude is mid-Phase-0 right now). On the Opus side:

- Adding **Exp 008: In-Place TTT** — new TTT objective + MLP-`down_proj`-only filter. Independent from selective-TTT (Exp 002) so they can run in parallel without crossing wires.
- Adding **Exp 009: qTTT** — `q_only` filter + cached K/V eval pattern. New env vars: `TTT_OBJECTIVE ∈ {ce, ntp_outer}`, `TTT_CACHE_KV ∈ {0,1}`.
- Treating Mousse / NuMuon / BHyT as Day-3 reserve only — they need full retrains (3-seed × 8×H100) and the SOTA's Muon/RMSNorm hyperparams may not transfer.
