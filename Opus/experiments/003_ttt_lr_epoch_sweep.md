# Experiment 003 — TTT LR × epochs sweep on the winning filter

**Date:** TBD (Day 2)
**Hypothesis:** With a smaller adapt surface (e.g. `scales`), the optimal `TTT_LR` is higher than 0.005 and the optimal `TTT_EPOCHS` may differ from 3. Selective-TTT changes the loss landscape — the SOTA's tuning was for `all`.
**Baseline:** The winning filter from Experiment 002 at SOTA defaults (LR=0.005, EPOCHS=3)
**Cost:** ~5×3 grid = 15 cells × ~2 min × $6/hr = ~$3 per single seed

## Setup

Run only after Experiment 002 identifies a winning filter. Use the saved checkpoint from Day 1.

## Grid

```
TTT_LR     ∈ {0.002, 0.005, 0.010, 0.020, 0.040}
TTT_EPOCHS ∈ {1, 3, 5, 7}
```

20 cells. Each ~2 min on 2×H100 (eval-only via `EVAL_ONLY=1` reuses the Day-1 checkpoint). Total ~40 min at $6/hr ≈ $4.

For `scales` filter specifically — since the gradient flow is now restricted to ~38K floats — much higher LRs (e.g. 0.04, 0.1) may work; we extend the LR grid upward only after seeing whether smaller LRs already saturate. Likewise, with a 1000× smaller adapt surface, more epochs (5, 7) are cheap to test (eval cost grows linearly with epochs but selective TTT skips backward through quantized matrices, which is the bulk of full-param TTT cost).

### Recon-driven additions

From mining the records leaderboard ([Explore agent report, 2026-04-27]):
- The TTT lineage on the leaderboard has **never** ablated TTT_EPOCHS above 3 — uncharted territory.
- TTT_LR has only been measured at 0.002 and 0.005. {0.010, 0.020, 0.040} are all unmapped.
- TTT_MOMENTUM has been pinned at 0.9 across every published TTT submission. {0.0, 0.95} are unmapped.

## Commands

```bash
WINNER_FILTER=scales   # set from experiment 002
export CKPT=/workspace/artifacts/seed42.int6.ptz

for LR in 0.002 0.005 0.010 0.020 0.040; do
  for EP in 1 3 5 7; do
    TAG="lr${LR//./_}_ep${EP}"
    EVAL_ONLY=1 TTT_ENABLED=1 TTT_PARAM_FILTER=$WINNER_FILTER \
      TTT_LR=$LR TTT_EPOCHS=$EP \
      SEED=42 LOAD_CHECKPOINT=$CKPT \
      RUN_ID=opus_e003_${TAG} \
      torchrun --standalone --nproc_per_node=2 \
        Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/003_${TAG}.log
  done
done
```

## Result

Fill in:

| LR \ EPOCHS | 1 | 3 | 5 | 7 |
|-------------|---|---|---|---|
| 0.002       |   |   |   |   |
| 0.005       |   |   |   |   |
| 0.010       |   |   |   |   |
| 0.020       |   |   |   |   |
| 0.040       |   |   |   |   |

## Decision

Pick the LR×EPOCHS combo with the lowest `val_bpb_ttt`. If multiple are within 0.0003: pick the one with the lower epochs count (faster eval, more headroom for the 600s eval budget).

If best cell beats Experiment 002's `f_all` baseline by **≥0.003 nats**: promote to Experiment 004 (chunk-size sweep on the winning config).
