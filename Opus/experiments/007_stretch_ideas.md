# Experiment 007 — Stretch ideas (Day 3 reserve, only if mainline wins)

**Date:** TBD (Day 3 morning, only if Exp 002–006 produce a stable winner ≥0.003 nats above SOTA single-seed)
**Hypothesis:** Several cheap, untried directions found in the records-folder recon stack on top of selective TTT.
**Baseline:** The locked Day-2 winner (filter + LR/epochs + 2nd-order knobs).
**Cost:** Each cell is a **full retrain** (since these change architecture/training, not just eval). Budget: ~$30 if all four cells, single seed. Skip if Day 2 winner is marginal.

## Why this exists (recon-driven)

From the leaderboard recon report (Opus/notes/recon_records.md):

1. **QK_GAIN_INIT was monotonically improving** with each step from 4.0 → 5.0 → 5.25 (current SOTA). 5.5+ has never been tested. **Could give another −0.0001 to −0.0005 nats for free.**
2. **TTT only ever sees the trained model**, not an EMA-averaged version separately. The EMA decay 0.9965 is applied at end of training; a different decay value or none-at-end might pair better with selective TTT.
3. **`COMPRESSOR='brotli'`** is hardcoded; the script also has paths for `zstd`. Switching could reclaim a few hundred bytes from the artifact, freeing budget for finer GPTQ.
4. **No one has tried `MUON_BACKEND_STEPS` other than 5** — increasing to 6 or 7 may give cleaner Newton-Schulz convergence.

## Configs

| Tag | Variant | Cost (8×H100 retrain) |
|-----|---------|------------------------|
| `qk_55` | `QK_GAIN_INIT=5.5`, all else SOTA | ~$5 |
| `qk_575` | `QK_GAIN_INIT=5.75` | ~$5 |
| `qk_60` | `QK_GAIN_INIT=6.0` | ~$5 |
| `ema_off` | `EMA_DECAY=0` (no EMA) | ~$5 |
| `ema_996` | `EMA_DECAY=0.9960` | ~$5 |
| `muon_bs_6` | `MUON_BACKEND_STEPS=6` | ~$5 |

Pick whichever 2 look highest-EV after seeing Exp 002–006. Don't run all 6.

## Commands

```bash
# QK_GAIN sweep (most important)
for QK in 5.5 5.75 6.0; do
  TAG="qk_${QK//./_}"
  TTT_ENABLED=1 TTT_PARAM_FILTER=$WINNER_FILTER \
    TTT_LR=$WINNER_LR TTT_EPOCHS=$WINNER_EP TTT_CHUNK_TOKENS=$WINNER_CHUNK \
    QK_GAIN_INIT=$QK \
    SEED=42 \
    RUN_ID=opus_e007_${TAG} \
    torchrun --standalone --nproc_per_node=8 \
      Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/007_${TAG}.log
done
```

## Decision

- Best stretch cell beats Day-2 winner by ≥0.001 single-seed → fold it into the 3-seed validation (Exp 005).
- No cell wins → ship Day-2 winner unchanged.

## What we are NOT doing

- **BigramHash, SmearGate** — different code paths entirely (different model class). Code-budget cost is ~3KB compressed, hard to fit. Documented for later.
- **MLA / MTP** — full architectural rewrites; weeks of tuning.
- **Mixed-bit GPTQ** — already pre-spec'd as Exp 010 fallback; only if mainline TTT angle dies.
- **ETLB / SLOT** — explicitly off in the SOTA submission as compliance requirements. Don't touch.
