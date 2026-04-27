# Records-folder recon (2026-04-27)

Distilled from an Explore-agent pass over `records/track_10min_16mb/` (32 record submissions) and `records/track_non_record_16mb/` (4 non-record). Each summary keeps actionable info; the full agent transcript is in conversation memory.

## Headline finding: selective-param TTT is genuinely unexplored

> **No submission has attempted selective-parameter test-time training restricting adaptation to a small control surface.**

- The only "freeze" knob anyone tried is `TTT_FREEZE_BLOCKS`, which freezes whole layers. PR `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` ablation: `freeze=2` cost −0.0004 nats vs `freeze=0` (i.e. freezing late layers slightly hurt).
- Nobody has tested "freeze every weight matrix and adapt only the ~38K control-surface floats."
- All TTT submissions use full-model SGD on 34M params.

This is the angle. PR #1493's TTT gain is −0.0027 nats; if selective TTT can extract another −0.005 we beat SOTA.

## TTT lineage with measured deltas

| Date | Submission | TTT Variant | Δ vs prior | Result |
|------|-----------|-------------|-----------|--------|
| 2026-03-17 | LoRA TTT (sam) | per-doc rank-8 LoRA, 1 Adam step | +0.0286 (worse) | proof-of-concept |
| 2026-03-23 | LeakyReLU_LegalTTT | full SGD lr=0.002 ep=3 freeze=0 | −0.0025 | first legal TTT on the leaderboard |
| 2026-04-06 | SP8192_QK5_LegalTTT | full SGD lr=0.005 ep=3 freeze=0 score-first | −0.0028 | sets 1.0828 |
| 2026-04-08 | ParallelResid_ScoreFirstTTT | same TTT + parallel residuals layers 7+ | −0.0022 | 1.0822 |
| 2026-04-09 | 3LayerRecur_ParResid_QK525 | same TTT + 3-layer depth recurrence | −0.0027 | **current SOTA 1.0810** |

TTT delta is reliably −0.0015 to −0.003 nats. Not a moonshot — a steady contribution layered onto whatever base trains best.

## Sweeps the leaderboard has NEVER run

- **`TTT_EPOCHS`**: locked at 3. No public ablation of 1, 2, 4, 5, 7, 10. Selective TTT makes more epochs cheap (1000× smaller surface, no backward through quantized matrices).
- **`TTT_LR`**: only 0.002 and 0.005 ever measured. {0.001, 0.010, 0.020, 0.040, 0.100} all unmapped.
- **`TTT_MOMENTUM`**: pinned at 0.9. No-momentum and high-momentum (0.95) untested.
- **`TTT_CHUNK_TOKENS`**: locked at 32K. {8K, 16K, 64K, 128K} untried.
- **Momentum reset between chunks**: never tested.
- **Weight decay during TTT**: never tested.
- **LR schedule shape (cosine vs constant vs linear)**: cosine baked in, others unexplored.

→ Exp 003, 004, 006 cover this ground.

## Cheap wins not yet stacked into SOTA

- **`QK_GAIN_INIT` > 5.25**: monotonic improvement 4.0 → 5.0 → 5.25 across submissions; 5.5+ never tested.
- **`MUON_BACKEND_STEPS` > 5**: never tested (default 5 in every submission).
- **EMA decay other than 0.9965**: rarely tuned post-PR-#1394.
- **GPTQ block size other than 128**: never swept.

→ Exp 007 picks the highest-EV of these as a Day-3 stretch.

## Disabled-but-wired flags in the SOTA code

These have code paths but default to off. Most are off because the PR #1493 author judged them either redundant or risky.

| Flag | Path | Risk |
|------|------|------|
| `ETLB_ENABLED` | Eval-time logit bias | **Compliance violation** (Issue #1017 #2). Don't touch. |
| `SLOT_ENABLED` | Scored-position lookup | **Compliance violation**. Don't touch. |
| `MTP_HEADS` | Mixture-of-experts heads | Untuned; significant byte cost. Skip unless desperate. |
| `BIGRAM_VOCAB_SIZE` (with hash) | Bigram embedding cache | Used through ~2026-03-29; dropped. Could pair with selective TTT. |
| `SWA_ENABLED` | Stochastic weight averaging | Tuned per-run, last clear use 2026-03-20. |
| `LATE_QAT` | Quantization-aware training near end of train | Never adopted in SOTA stack. |

→ Exp 010 (mixed-bit GPTQ) is our planned non-TTT fallback. BigramHash is interesting but adds compressed-code cost we don't have headroom for.

## What we are explicitly NOT pursuing (and why)

- **Architecture rewrites (MLA, ternary weights, MTP)**: weeks of tuning, days remaining.
- **ETLB / SLOT**: rule-violating per the SOTA's own compliance manifest.
- **BigramHash + selective TTT combo**: code-budget too tight (we already used +528 lzma bytes for v1; SOTA's smallest seed has only 8070 bytes total slack).
- **Antigravity stack (MLA, 3.5× MLP, Int6 QAT)**: separate non-record submission, not the leaderboard angle.

## Code-budget accounting (lzma compressed)

- `train_gpt_base.py` (PR #1493 SOTA): 13,248 bytes lzma → ~16,560 bytes b85-wrapped
- `train_gpt_v1.py` (Opus selective-TTT patch): 13,776 bytes lzma → ~17,220 bytes b85-wrapped
- **Δ = +528 bytes lzma**

PR #1493 artifact bytes per seed: 15,991,930 / 15,992,919 / 15,993,232. Worst-case slack is 16,000,000 − 15,993,232 = 6,768 bytes. After our +528 lzma: ~6,240 bytes. Workable but tight; for the final submission we strip the verbose log line and prune unused schedule branches if needed.
