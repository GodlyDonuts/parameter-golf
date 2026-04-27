# Experiment 006 — TTT 2nd-order knob sweep

**Date:** TBD (Day 2, after Exp 003 picks the winning LR×EPOCHS)
**Hypothesis:** With the winning filter + LR + epoch combo locked, the next-tier TTT knobs (`MOMENTUM_RESET`, `WD`, `SCHEDULE`, `LR_OFFSET`) each unlock 0.0005–0.001 nats individually and may stack.
**Baseline:** Best cell from Experiment 003.
**Cost:** ~10 cells × ~2 min = ~$2 (eval-only, single seed)

## Why this exists

The Explore-agent recon (Opus/notes/recon_records.md, summarized in DECISIONS.md 2026-04-27) found that the published TTT lineage has **never** swept these knobs:
- Momentum reset between chunks — momentum carries by default; resetting may reduce inter-chunk drift.
- Weight decay — never tested for TTT; theoretically helps full-param filters fight quantization-grid memorization.
- Cosine vs constant vs linear LR schedule across chunks.
- LR offset — shifts peak LR away from chunk 0. Less defensible (chunk 0 has the most downstream chunks to influence) but cheap to test.

These are orthogonal to LR/epochs, so we sweep one-at-a-time vs the Exp-003 winner.

## Configs

All on the Day-1 checkpoint with `EVAL_ONLY=1`. Substitute `$WINNER_*` from Exp 003.

| Run ID            | Variant | Notes |
|-------------------|---------|-------|
| `b_winner`        | reproduce Exp 003 best | reference |
| `mom_reset`       | `TTT_MOMENTUM_RESET=1` | clear SGD momentum between chunks |
| `mom_0`           | `TTT_MOMENTUM=0.0` | no momentum at all |
| `mom_95`          | `TTT_MOMENTUM=0.95` | hotter momentum |
| `wd_001`          | `TTT_WD=0.001` | tiny weight decay |
| `wd_01`           | `TTT_WD=0.01` | larger weight decay |
| `sched_constant`  | `TTT_SCHEDULE=constant` | flat LR across chunks |
| `sched_linear`    | `TTT_SCHEDULE=linear` | linear decay |
| `lr_offset_1`     | `TTT_LR_OFFSET=1` | peak LR shifts to chunk 1 |
| `lr_offset_2`     | `TTT_LR_OFFSET=2` | peak LR shifts to chunk 2 |

## Commands

```bash
export CKPT=/workspace/artifacts/seed42.int6.ptz
COMMON="EVAL_ONLY=1 TTT_ENABLED=1 TTT_PARAM_FILTER=$WINNER_FILTER \
        TTT_LR=$WINNER_LR TTT_EPOCHS=$WINNER_EP TTT_CHUNK_TOKENS=$WINNER_CHUNK \
        SEED=42 LOAD_CHECKPOINT=$CKPT"

# Reference
eval $COMMON RUN_ID=opus_e006_b_winner torchrun --standalone --nproc_per_node=2 Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/006_b_winner.log

# Each variant
for SPEC in \
    "mom_reset:TTT_MOMENTUM_RESET=1" \
    "mom_0:TTT_MOMENTUM=0.0" \
    "mom_95:TTT_MOMENTUM=0.95" \
    "wd_001:TTT_WD=0.001" \
    "wd_01:TTT_WD=0.01" \
    "sched_constant:TTT_SCHEDULE=constant" \
    "sched_linear:TTT_SCHEDULE=linear" \
    "lr_offset_1:TTT_LR_OFFSET=1" \
    "lr_offset_2:TTT_LR_OFFSET=2"; do
  TAG=${SPEC%%:*}; EXTRA=${SPEC#*:}
  eval $COMMON $EXTRA RUN_ID=opus_e006_${TAG} \
    torchrun --standalone --nproc_per_node=2 Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/006_${TAG}.log
done
```

## Result

| Run ID | val_bpb_ttt | Δ vs b_winner | Notes |
|--------|-------------|---------------|-------|
| `b_winner` | | 0 | reference |
| `mom_reset` | | | |
| `mom_0` | | | |
| `mom_95` | | | |
| `wd_001` | | | |
| `wd_01` | | | |
| `sched_constant` | | | |
| `sched_linear` | | | |
| `lr_offset_1` | | | |
| `lr_offset_2` | | | |

## Decision

- Any individual knob beats `b_winner` by ≥0.0005 nats: keep it for stacking.
- Stack 2–3 winners and one final eval to confirm joint gain.
- If joint stack beats `b_winner` by ≥0.001 nats: lock in for Exp 005 (3-seed validation).
- If nothing helps: ship Exp-003 winner without 2nd-order knobs.
