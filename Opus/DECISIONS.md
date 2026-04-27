# Decision Log

Audit trail for non-obvious calls. Each entry: date, decision, alternatives considered, reasoning. New entries at top.

---

## 2026-04-27 — Selective-TTT angle confirmed novel; layered TTT-knob sweep added

**Decision:** Stay on the selective-TTT primary hypothesis, add `EVAL_ONLY` mode + 5 second-order TTT knobs (momentum reset, wd, grad clip, lr offset, schedule) to `train_gpt_v1.py`, and add Exp 006 (2nd-order knob sweep) and Exp 007 (Day-3 stretch ideas).

**Inputs to this decision:**
- Explore-agent recon over `records/` (32 records + 4 non-records). Headline: **no submission has ever attempted selective-parameter TTT**. The only freeze-related ablation was whole-layer freezing in 2026-03-23 (`freeze=2` cost −0.0004 nats vs `freeze=0`), which is coarse-grained and unrelated to the ~38K-float fp32 control surface.
- TTT lineage gains are −0.0015 to −0.003 nats reliably. Selective TTT needs to extract another −0.005 to clear the bar.
- The leaderboard has never swept `TTT_EPOCHS` above 3, never tested momentum reset, never tested weight decay during TTT, never compared cosine vs constant vs linear schedules.
- Code-budget after the v1 patch: +528 lzma bytes vs SOTA. Worst-seed artifact slack drops from 6,768 → 6,240 bytes — still positive.

**Alternatives considered:**
- (A) Add BigramHash + selective TTT — interesting but adds another ~3KB compressed; we're already at +528 bytes and only ~6K slack on the worst seed.
- (B) Pivot now to mixed-bit GPTQ — premature; selective-TTT is unproven but cheap to test (eval-only).
- (C) Pursue stretch architecture (MLA / MTP) — weeks of work; 3 days remaining.
- (D) Stay on plan, layer in 2nd-order TTT knobs as orthogonal one-knob-at-a-time sweeps. **← chosen**

**Reasoning:** The primary angle is novel and cheap to validate; doubling down with orthogonal knob sweeps maximizes the chance of finding the −0.005 we need. Each new knob defaults to no-op so byte-for-byte SOTA reproduction is unchanged.

---

## 2026-04-27 — EVAL_ONLY mode + LOAD_CHECKPOINT support

**Decision:** Add an `EVAL_ONLY=1` env var that skips `train_model + serialize` and goes straight to `deserialize → sliding eval → TTT eval`, plus a `LOAD_CHECKPOINT` env var that overrides `quantized_model_path`.

**Reasoning:** Day-2 TTT sweeps need to iterate ~30+ configs. A full 600s train + ~500s eval cycle is $4–5 per cell on 8×H100. Eval-only on a saved Day-1 artifact is ~$1 per cell. Borrowed pattern from Claude/scripts/eval_only_patch.md (sibling agent's converging design).

---

## 2026-04-27 — Build on PR #1493 SOTA, not on `train_antigravity.py`

**Decision:** Use the SOTA file at `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` as the base for the leaderboard push. Treat `train_antigravity.py` as a separate, parallel non-record submission.

**Alternatives considered:**
- (A) Push only the antigravity stack (MLA + 3.5× MLP + Int6 QAT, vocab 1024) for the record.
- (B) Bolt antigravity ideas (MLA, MTP) onto the SOTA stack from scratch.
- (C) Build directly on PR #1493 SOTA, keep antigravity as a side submission. **← chosen**

**Reasoning:**
- The antigravity stack is missing every layer of the current SOTA (no SP8192, no GPTQ-SDClip, no recurrence, no parallel residuals, no legal TTT). Reaching parity is weeks of work, not 3 days.
- The SOTA author chain is on the same code surface — adding to it is incremental and stylistically expected by the reviewers.
- Keeping antigravity alive as a non-record submission is ~free in compute and gives us a guaranteed submission either way.

---

## 2026-04-27 — Top angle: smarter TTT, not more architecture

**Decision:** Spend the bulk of compute exploring TTT variants (param-selective, chunk-size sweeps, momentum schedules) rather than architectural changes (MLA, MTP, depth changes).

**Alternatives considered:**
- New attention variant (MLA / linear / state-space hybrid) — requires retraining from scratch and weeks of tuning.
- More depth recurrence loops — already at 3 layers × 2 loops; diminishing returns.
- New optimizer (Lion, Sophia) — Muon is hard to beat at this scale.
- Mixed-bit GPTQ — kept as fallback (Day 2 pivot).

**Reasoning:**
- TTT is the newest layer in the SOTA stack (added in PR #549, refined through #1413, #1493). Less time for the community to optimize it.
- Current TTT is naïve: vanilla SGD on **all** params. A quantized model has only a small fp32 surface (`q_gain`, `attn_scale`, `mlp_scale`, `skip_weights`, `skip_gates`, `resid_mix`, `ln_scale_factor`); training only those is faster, lower-variance, and avoids fighting GPTQ rounding errors.
- TTT runs at eval time, so we can iterate on a fixed checkpoint without re-paying the 10-min training cost per experiment. This makes Day 2 triage 10× cheaper than retraining sweeps.
- Theoretical ceiling: TTT currently gives ~0.002 nats (1.0827 sliding → 1.0810 TTT). If we can extract another 0.005 from the same mechanism, that's our submission.
