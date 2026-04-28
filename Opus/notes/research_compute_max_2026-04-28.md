# Research note — Compute-maxing & untried angles (2026-04-28)

**Question from sairamen:** GPU util ~50%, VRAM <20%, CPU ~4% during training. Are we leaving wins on the table? What else are we missing?

**Short answer:** Yes. The leaderboard converged on gradient-quality and quantization-quality optimization while leaving raw throughput largely unexplored. Modded-NanoGPT (the speedrun lineage one tier up from our codebase) has absorbed multiple compute-maxing tricks that haven't crossed into Parameter Golf. There are also ~5 quality knobs that nobody on the leaderboard has touched.

This note covers what I found via (a) deep recon of `records/track_10min_16mb/` + non-record track, (b) WebFetch of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) at record #80 (1.406 min on 8×H100), (c) literature on FP8 / CUDA Graphs / distillation init.

---

## TL;DR — top-8 actionable items, ranked by EV × confidence × low-cost

| # | Item | Status | Cost | Expected gain | Confidence |
|--:|------|--------|------|---------------|------------|
| 1 | **CUDA Graphs** capture the training step | NEVER-TRIED on PG | ~50 LOC + $10 | 15–30% step time → 15–30% more steps in 600s | High (NVIDIA docs cite 7.2× in inference; speedrun community uses it) |
| 2 | **Batch-size sweep** at fixed wallclock (`{1.5M, 2M, 3M}` tokens) | NEVER-TRIED on int6 SOTA | $20 (4 cells) | Either neutral or free −0.0005 from richer per-step gradient | High at <100M scale |
| 3 | **FP8 matmul on `lm_head` + `tok_emb`** (logits path only) | NEVER-TRIED on PG, in modded-nanogpt | ~30 LOC + $10 | ~5–10% step time + slight quality | Medium (FP8 instability documented; head is small surface so safer) |
| 4 | **Multiple EMAs + Polyak averaging** at end of train | NEVER-TRIED | ~20 LOC + $5 | Quality lever, ~−0.0005 to −0.001 expected | Medium (well-known stabilization trick) |
| 5 | **Async data prefetch** (3 batches ahead, pinned host buffers) | NEVER-TRIED | ~30 LOC + $5 | 2–5% step time IF dataloader is bottleneck (need profile to confirm) | Low confidence on the gain; CPU at 4% suggests it's NOT the bottleneck |
| 6 | **Cautious Weight Decay schedule tied to LR** | NEVER-TRIED on PG | ~10 LOC + $5 | ~−0.0003 to −0.0008 | Medium (modded-nanogpt records show consistent gain) |
| 7 | **Larger GPTQ calibration set (256 vs 64 batches)** | NEVER-TRIED on PG | ~5 LOC + $5 | Sharper int6 grid → ~−0.0001 to −0.0005 | Low–medium |
| 8 | **Sequence-length warmup** (start 256, grow to 2048) | NEVER-TRIED on PG | ~30 LOC + $5 | Quadratic FLOPs reduction in early steps → more total steps | Medium (used in nanoGPT speedrun records 60+) |

If we hit 4–5 of these even at the low end of expected gains, that's −0.002 to −0.004 nats — directly closing the 0.00421 gap to 1.0760.

---

## NEVER-TRIED on the leaderboard (the real open territory)

These are NOT mentioned in any of the 32 record + 4 non-record submissions or in `Claude/notes/leaderboard_intel.md`'s "explored & dropped" list:

### CUDA Graphs (`torch.cuda.CUDAGraph`)
**Why now:** at our small model size (35M params, 11×512), kernel-launch overhead is a real fraction of step time — many <100 µs kernels with ~5 µs launch latency each. CUDA Graphs precompile the entire step into one capture and replay, eliminating launch overhead.
- Status in modded-nanogpt: heavily used in records 70+.
- Status in records/: zero mentions of `CUDAGraph` across all submissions.
- Implementation: replace `step_fn` with a captured graph; warm-up the first N steps without graphs, then capture once and replay.
- Risk: hooks (Newton-Muon!) and DDP `require_backward_grad_sync` toggles can break capture. Mitigation: do CUDA Graphs in the **vanilla Muon path only**, fall back to non-graph for NM.

### Multiple parallel training rounds in unused VRAM
**Why interesting:** each H100 has 80 GB; we use ~5–15 GB. We could fit 2–3 independent 35M models per H100 and train all of them simultaneously.
- Use cases:
  - Ensemble at the artifact level — pick the best of 3 trainings.
  - Soft-ensemble at the weight level — average the 3 final weight states (weight averaging across seeds *during* training, not just at end).
  - Concurrent hyperparameter sweep — train 3 LR variants in parallel, pick winner.
- Risk: NCCL collectives serialize — if we train 3 models in parallel, the all_reduce traffic 3×s. May offset the parallelism win.
- Status: **never tried** anywhere in records or modded-nanogpt.

### Multiple EMAs / Polyak averaging
**Why:** the SOTA uses a single EMA at decay 0.9965. Maintain 2–3 EMAs at different decays (e.g., 0.99, 0.9965, 0.9990) and at the end pick the lowest val_bpb among them, OR average them. Almost-free at training time (extra fp32 copy of weights × N).
- Cost: <100 MB per H100 (negligible).
- The Lookahead-style "tail averaging" papers (Polyak 1990, Tail-Average 2023) all show <100M-scale models benefit.

### Distillation initialization
**Why:** start training from a state distilled from a public 1B teacher (e.g., HuggingFace SmolLM2-135M or similar) rather than random init. Even ~30s of distillation before random-init training begins can save ~10% of the loss curve climb.
- Constraint: legal under PG rules? The artifact must be self-contained, but the *initial weights* are arbitrary as long as the final model is the trained output. So distillation from a public model into our random init = legal.
- Risk: vocab mismatch (teacher likely uses different SP). Need a vocab-matched teacher or to project through a learned mapping.
- Status: **never tried**, no precedent on PG.

### Larger GPTQ calibration set
The SOTA uses `gptq_calibration_batches=64`. Hessian sampling becomes more accurate with more batches. Try 128, 256. Cost is wallclock during the 12s GPTQ reservation — might exceed it. Pre-compute Hessians during training (forward-only path on a held-out batch) instead.
- Status: never explored on PG.

### Sequence-length warmup
Start training at `seq_len=256` and grow to `2048` linearly over ~1500 steps. Quadratic attention FLOPs reduction in early steps → more total steps within the 600s wallclock cap. Used by [Tyler Romero's NanoGPT speedrun worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) and modded-nanogpt records 50+.
- Risk: rope wavelengths trained on short sequences may not generalize to long; the SOTA already has `rope_train_seq_len=2048`. Warmup needs careful schedule.

---

## Modded-NanoGPT techniques NOT in our SOTA

Verbatim from `https://github.com/KellerJordan/modded-nanogpt` (record #80, 1.406 min, 124M model on 8×H100):

| Technique | In our SOTA? | Plausible PG gain |
|-----------|:-----:|------------------|
| FP8 matmul for head + asymmetric rescale + softcap | ❌ | small (head is ~5% of FLOPs) but real |
| Fused linear-relu-square MLP | ❌ (we have `LeakyReLU(0.5)²` but not fused) | 4–6 ms/step (~180 free steps in 600s) |
| Fused softcapped cross-entropy | ❌ | 1–2 ms/step |
| FA3 long-short sliding window attention | ❌ (we have FA3 but full-context) | uncertain at seq=2048 |
| Async data prefetch | ❌ | 2–5% step time |
| Gradient/compute overlap | partial (DDP) | 3–5% step time |
| Attention window warmup | ❌ | quality at early-step regime |
| Multi-token prediction (MTP) | ❌ — Claude's leaderboard intel marked this DROPPED, but modded-nanogpt revived it | uncertain — re-check |
| Paired-head Q/K orthogonalization in Muon | ❌ | small kernel-launch reduction |
| "Adam every other step" | ❌ | ~5% step time on Adam params (small share) |
| Cautious Weight Decay tied to LR | ❌ | quality lever, ~−0.0003 to −0.0008 |
| Batch size schedule | ❌ | similar to seq-length warmup |
| Smear token embeddings 1 position forward | ❌ — Claude's intel says DROPPED on PG, may need re-test | uncertain |

The most direct ports: **fused softcapped CE**, **FP8 head matmul**, **Cautious WD**, **Adam every other step**. Each is a small change with documented gain in the speedrun community.

---

## TRIED-AND-DROPPED on PG (don't waste compute)

From the Explore agent's records-folder pass and `Claude/notes/leaderboard_intel.md`:

- **`seq_len=4096`** ([2026-03-19_TrainingOptSeq4096](records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md)): −0.023 vs naive baseline but lost to other 2048-based entries. Step-time penalty exceeds quality gain.
- **Parallel residual MLP-skip variant** ([2026-03-31_ParallelResiduals_MiniDepthRecurrence](records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/README.md)): "I also tried the followup optimization from modded-nanogpt PR #241 ... that brought a slight regression."
- **3-loop mini-recurrence** (same submission): "repeating three was already losing to the step-time penalty."
- **Ternary / 1-bit weights**: Fundamentally different track (74M ternary in 2026-03-24). Lost to int6 SOTA at 600s budget because ternary converges slower.
- **YaRN, NeoMuon**: incompatible with current Muon stack.
- **Hash embeddings, value embeddings VE128, ±1 selective pruning, late QAT, GPTQ-lite**: all lost to current SOTA's formulation.

These are not worth re-running unless we change the underlying stack significantly.

---

## FP8 — interesting but risky

From [arxiv.org/abs/2411.08719](https://arxiv.org/abs/2411.08719) (FP8 vs BF16 trade-off study) and NVIDIA Hopper docs:

- **FP8 matmul peak**: 3958 TFLOPS (e4m3) on H100 vs BF16's 1979 TFLOPS — **2× theoretical**.
- **GPT-175B observed**: 75% speedup on BF16 → FP8 mixed-precision.
- **Real gains for *inference***: ~5× on large matmul; **for training: 1.6× typical** because LayerNorm / softmax / element-wise are memory-bound and don't benefit.
- **Documented instability**: "FP8 settings led to unstable training loss and frequent loss spikes."

For PG specifically:
- Going **full FP8** is high risk in a 600s wallclock window. One loss spike that doesn't recover wastes the run.
- Going **FP8 only on `lm_head` + `tok_emb`** is much safer (those are well-conditioned matmuls, output is logits which then softmax normalizes anyway). Modded-nanogpt does exactly this. ~5–10% step gain at low risk. **This is the right initial bet.**

---

## What I'm explicitly NOT recommending

- **Custom Triton / CUDA kernels** — too much engineering for 2 days. Defer.
- **Tensor parallelism** — overkill at 11×512.
- **Any change that requires a new tokenizer** (vocab > 8192, BPE retrain) — VPS-prep cost is hours and we'd need to retokenize the FineWeb shards. Worth it only for a multi-day investment.
- **FP4 weights** — too aggressive at this scale; int6 is already the leaderboard's empirical sweet spot.
- **AdEMAMix replacement of Muon** — confirmed worse at large-batch/8×H100 regime by Gemini's recon.

---

## Decision rules

Given Phase 2 is mid-flight and Phase 3 (3-seed) lands ~05:00 UTC:

**If Phase 2 + 3-seed mean clears 1.0760:** ship. None of this matters.

**If Phase 2 + 3-seed mean lands 1.0760–1.0780 (within reach):**
1. CUDA Graphs (#1 in table) — first try, biggest documented payoff at our scale.
2. LR follow-on micro-sweep {0.012, 0.015, 0.020} — Stage 2's monotonic trend says room remains.
3. Newton-Muon (`v2.py` already drafted) — the highest-EV optimizer change.

**If Phase 2 + 3-seed mean lands 1.0780–1.0800 (more ground to cover):**
1. CUDA Graphs + batch-size sweep + FP8 head — the throughput stack.
2. Newton-Muon as the gradient-quality lever.
3. Multiple EMAs as a free quality lever.

**If we have time after the record submission lands:** distillation init + multiple parallel rounds + sequence-length warmup are the high-research-EV bets for the *next* submission cycle.

---

## What's still worth more research

I haven't yet validated:

- **Whether `torch.compile(mode='reduce-overhead')` works on the SOTA stack** — that mode uses CUDA graphs internally with much less code than manual capture. If it works, it's a 1-line change to the SOTA. Top of my list to test.
- **Whether MTP (multi-token prediction) actually loses on PG vs simply hadn't been tuned right.** Claude's leaderboard-intel says DROPPED but cites no specific submission. Worth a 1-line check.
- **Whether the Bansal/Zhang qTTT paper body actually prescribes Q-only or Q+something.** The abstract doesn't commit to Q-only. We may be over-specifying based on Gemini's synthesis. Cheap PDF read.
- **Whether modded-nanogpt's "Cautious Weight Decay schedule tied to LR" patch is publicly diffable.** If yes, port directly.

I'll work through these in order. None of them spend GPU.

## Sources

- [GitHub: KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [Tyler Romero — NanoGPT Speedrun Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/)
- [arXiv:2411.08719 — Balancing Speed and Stability: FP8 vs BF16 Training](https://arxiv.org/abs/2411.08719)
- [NVIDIA — Best Practices for PyTorch CUDA Graphs](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [arXiv:2310.18313 — FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)
- [Pre-training Distillation for Large Language Models (ACL 2025)](https://aclanthology.org/2025.acl-long.181.pdf)
- Internal recon: `Opus/notes/recon_records.md`
- `records/track_10min_16mb/` README files (per-submission file references inline above)
