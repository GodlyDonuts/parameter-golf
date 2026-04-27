# Experiment 002 — TTT param-filter sweep

**Date:** TBD (Day 2)
**Hypothesis:** Restricting TTT to the fp32 control surface (`scales`) beats `all` at fixed compute by adapting the post-GPTQ residual error without fighting the quantization grid.
**Baseline:** Experiment 001 reproduction (TTT on `all`, expected ≈ 1.0808)
**Cost:** 6 configs × ~2 min eval each on 2×H100 = ~12 min × $6/hr = ~$1.20 per single seed; budget ~$15 with rerun headroom

## Why this exists

The current TTT updates every parameter (~34M floats) on a 32K-token chunk × 3 epochs. The matrix params just came off a careful GPTQ rate-distortion fit — every TTT step pulls them off that grid for marginal data signal. Hypothesis: only the fp32 control surface (~38K floats) gives genuinely informative gradients on 32K tokens.

## Configs

All other hyperparameters held at PR #1493 SOTA values. Only `TTT_PARAM_FILTER` varies. We run on the **same checkpoint** (loaded from `final_model.int6.ptz`) so any difference is purely attributable to the TTT filter.

| Run ID | `TTT_PARAM_FILTER` | Floats updated | Expected sign |
|--------|---------------------|----------------|---------------|
| `f_all` | `all` | 34.5M | reference (reproduces 001) |
| `f_scales` | `scales` | ~38K | **primary hypothesis** — should ≈ tie or beat |
| `f_scales_embed` | `scales+embed` | ~4.2M | scales + tok_emb adaptation |
| `f_last3` | `last_n_layers:3` | ~7M | "fine-tune the head" approach |
| `f_attn` | `attn_only` | ~7M | attention-only adaptation |
| `f_mlp` | `mlp_only` | ~23M | MLP-only adaptation |

## Code

The patched script is `Opus/code/train_gpt_v1.py`. Same env-var surface as the SOTA plus:
- `TTT_PARAM_FILTER` (default `all`) — primary knob
- `EVAL_ONLY` (default `0`) — skip training, deserialize checkpoint, eval only
- `LOAD_CHECKPOINT` (default `final_model.int6.ptz`) — explicit checkpoint path
- `TTT_MOMENTUM_RESET`, `TTT_WD`, `TTT_GRAD_CLIP`, `TTT_LR_OFFSET`, `TTT_SCHEDULE` — orthogonal TTT knobs

All defaults match PR #1493 byte-for-byte.

## Commands

```bash
# Day-1 checkpoint copied to local $CKPT
export CKPT=/workspace/artifacts/seed42.int6.ptz

for FILTER in all scales scales+embed last_n_layers:3 attn_only mlp_only; do
  TAG=${FILTER//:/_}; TAG=${TAG//+/_}
  EVAL_ONLY=1 TTT_ENABLED=1 TTT_PARAM_FILTER=$FILTER \
    SEED=42 LOAD_CHECKPOINT=$CKPT \
    RUN_ID=opus_e002_${TAG} \
    torchrun --standalone --nproc_per_node=2 \
      Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/002_${TAG}.log
done
```

## Result

Fill in after running:

| Run ID | `val_bpb_ttt` | Δ vs `f_all` | Eval time | Notes |
|--------|---------------|--------------|-----------|-------|
| `f_all` | | 0 | | reference |
| `f_scales` | | | | |
| `f_scales_embed` | | | | |
| `f_last3` | | | | |
| `f_attn` | | | | |
| `f_mlp` | | | | |

## Decision criteria

- **`f_scales` beats `f_all` by ≥0.001 nats single-seed** → promote to Experiment 003 (LR sweep on `scales`)
- **`f_scales_embed` beats `f_all` by ≥0.001 nats** → keep as alt; sweep its LR separately
- **No filter beats `f_all`** → kill the selective-TTT angle, pivot to mixed-bit GPTQ (Experiment 010)
- **`f_scales` ties `f_all`** → still interesting (faster TTT, same quality); explore higher LR (since fewer params per gradient step, can take bigger steps)
