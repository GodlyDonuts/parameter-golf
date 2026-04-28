# Gemini Deep Research — Parameter Golf Literature Survey (2026-04-27)

**Source:** Gemini Deep Research, run by sairamen on 2026-04-27.
**Goal of the run:** survey literature for techniques that beat the standing SOTA `val_bpb = 1.0810` by ≥0.005 nats under the 16MB / 600s+600s constraints.
**Standing record at run time:** PR #1493 (bigbag, 2026-04-09).

> **Reader caveat:** This document records the verbatim Deep Research output. Several citations are dated 2025–2026 and were not independently verified at receipt time. Before committing GPU-hours to any technique, **fetch the cited arXiv ID and confirm the paper exists, the claim is real, and the regime is comparable to ours (≤100M params, ≤350M training tokens, int6 quant)**. The expected-BPB-delta numbers should be treated as upper-bound priors, not commitments.

---

## Top-10 Prioritized Techniques

| Rank | Technique | Type | Claimed Δ BPB | Implementation Cost | Notes |
|------|-----------|------|---------------|---------------------|-------|
| 1 | **In-Place TTT** | Eval-time | −0.004 to −0.008 | Low | NTP-aligned target on MLP final-projection only. Gemini's "most novel finding". |
| 2 | **Mousse Optimizer** | Train-time | −0.006 to −0.008 | Low | Shampoo-style preconditioner before Newton-Schulz. Drop-in for Muon. |
| 3 | **qTTT** (Query-only TTT) | Eval-time | −0.003 to −0.010 | Low | Cache K/V once; only Q gets gradients. |
| 4 | **CERWU** | Quantization | −0.003 to −0.008 | Mod-High | Rate-distortion optimal weight updates → smaller artifact, more capacity. |
| 5 | **BHyT Normalization** | Architecture | −0.004 | Low | Bounded tanh, 15.8% faster than RMSNorm → more steps in 600s. |
| 6 | **Progressive Depth Recurrence** | Architecture | −0.001 to −0.002 | Trivial | Stagger looping activation; eliminates loss-spike at frac=0.35. |
| 7 | **NuMuon** | Train-time | −0.002 to −0.006 | Moderate | Nuclear-norm-constrained Muon → low-rank weights → better Brotli. |
| 8 | **EngramLite** | Architecture | −0.003 | High byte cost | Multi-order n-gram hash w/ sigmoid gate. ~1.3 MB unquantized. |
| 9 | **TCA-TBE** | Compression | −0.001 to −0.002 (indirect) | High | Bitmap encoding into Tensor Cores; saves ~10–15s eval init time. |
| 10 | **Derf Activation** | Architecture | −0.003 | Low | Rescaled Gaussian erf, replaces both LayerNorm and activation. Mut-excl with BHyT. |

## Single most novel finding (per Gemini)

**In-Place Test-Time Training with NTP Alignment** — Feng et al. 2026, claimed arXiv:2604.06169. Replaces generic SGD-on-NLL with an NTP-aligned outer-product update applied only to MLP `down_proj` matrices: `W^(i) = W^(i-1) + η · V̂_[i]^T · Z_[i]` where `V_LM = E[x_{t+1}]`. Chunk-parallel, no full backward, doesn't disturb GPTQ-quantized attention weights.

**Why this matters for us:** every TTT submission on the leaderboard runs full-model SGD against cross-entropy loss. In-Place TTT is qualitatively different: it changes both the *target* (NTP-aligned, not generic CE) and the *surface* (MLP final projection only, not all 34M params). It's in the same family as my "selective TTT on the fp32 control surface" hypothesis but operates on a different (smaller, more semantically loaded) tensor set.

## Combinations Gemini called out

| Combo | Synergy | Combined Δ |
|-------|---------|-----------|
| Mousse + NuMuon | aligned-geometry descent + low-rank weights → fast & compressible | −0.005 to −0.009 |
| In-Place TTT + qTTT | MLP fast-weights × Q-only updates → max FLOPs / no forgetting | −0.006 to −0.010 |
| Progressive Recur + BHyT | bounded variance prevents shock at recurrence onset | −0.004 to −0.007 |
| CERWU + ANS | rate-distortion grid → entropy-coded → Shannon-limit artifact | −0.005 to −0.008 |

## Explicitly drop (per Gemini)

- **AdEMAMix** — outperforms only at small-batch / noise-heavy; loses to Muon/Mousse at large-batch pretraining.
- **TTT-Linear / TTT-MLP (Stanford)** — requires replacing FA3 attention; too disruptive for 3-day timeline.
- **Attention sinks / register tokens** — designed for 100K+ context; useless at seq_len=2048.
- **xIELU activation** — lacks fast kernels; Python fallback blows the 600s budget.
- **Test-time LoRA (post-quant)** — fp matrices instantiated at eval time → slow + memory.

## Full per-technique analysis

### Area A — Test-time training and adaptation

#### A.1 In-Place TTT (claim: arXiv:2604.06169, Feng et al. 2026)
- **Mechanism:** Bypasses generic reconstruction; isolates MLP final projection as fast weights; NTP-aligned target. Chunk-parallel update via outer product `V̂_[i]^T · Z_[i]`.
- **Why here:** Score-first eval is sequential; standard TTT auto-encodes current tokens (doesn't improve causal reasoning). NTP-aligned matches BPB.
- **Cost:** <500 bytes compressed. No full-model backprop.
- **Risks:** aggressive lr could destabilize int6 weights. Anneal per chunk.
- **Compliance:** OK — sequential, score-first preserved.

#### A.2 qTTT (claim: arXiv:2512.13898, Barua et al. 2025)
- **Mechanism:** Single prefill caches K/V; only Q projections get gradients in subsequent passes.
- **Why here:** Full-model SGD underutilizes H100. Freeing K/V → 2–3× more adaptation epochs in same budget.
- **Cost:** Low. Freeze non-Q params, manage KV cache.
- **Risks:** Q-only may lack capacity for deep distribution shift.
- **Compliance:** OK.

#### A.3 LaCT (claim: arXiv:2505.23884)
- **Mechanism:** Document-sized chunk updates with local windowing; Muon as fast-weight optimizer at inference.
- **Why here:** Pushes GPU utilization at eval from <5% to ~70%.
- **Cost:** Moderate restructure of eval window.
- **Risks:** Severe overfitting on current chunk → harms next chunk.
- **Conflicts:** With qTTT.
- **Compliance:** OK if scored before update.

#### A.4 MASS (claim: arXiv:2603.03524, Zweiger et al. 2026)
- **Mechanism:** Bilevel meta-learning of TTT init + LR during pretraining. Eval-time TTT converges in 1 epoch.
- **Cost:** **High** — bilevel inner/outer in the 600s training budget.
- **Risks:** Halving primary gradient steps probably nets a loss.
- **Compliance:** OK.

### Area B — Quantization at int6 and below

#### B.1 CERWU (claim: arXiv:2505.18758, Conzelmann & Bamler 2025)
- **Mechanism:** Co-optimizes quantization grid + weights to minimize `D(W,Ŵ) − λ log P(Ŵ)`. Optimal Brain Surgeon variant. Targets entropy of quantized weights, not just MSE.
- **Why here:** SOTA's GPTQ minimizes output distortion only; CERWU explicitly trains for compressibility, freeing 1–3 MB.
- **Cost:** Moderate–High, but offline (outside 600s).
- **Pairs with:** NuMuon (low-rank → low-entropy) additively.
- **Compliance:** OK.

#### B.2 AQLM (arXiv:2401.06118, Egiazarian et al. 2024) — **independently verifiable, real paper**
- **Mechanism:** Multi-codebook residual vector quantization for LLM weights, optimized at output level.
- **Why here:** Embeddings stuck at int8; AQLM compresses to ~3 bits/param at higher fidelity than int4 scalar.
- **Cost:** High — embed an RVQ decoder kernel in the LZMA-wrapped Python.
- **Risks:** Decode logic byte-cost may exceed weight savings on the 8K-vocab embedding.
- **Compliance:** OK.

#### B.3 OptRot / ROSAQ (Yoon et al. 2024/2025)
- **Mechanism:** Orthogonal rotation (Hadamard / PCA) of weights+activations before quantization → smooths outliers.
- **Why here:** int6 is outlier-sensitive; rotation enables tighter SDClip.
- **Cost:** Low. Rotations fuse into adjacent linears → zero param overhead.
- **Risks:** May interfere with In-Place TTT's geometric assumptions on MLP projections.
- **Compliance:** OK.

### Area C — Architectural micro-changes

#### C.1 BHyT Normalization (claim: arXiv:2601.09719, Byun et al. 2026)
- **Mechanism:** `BHyT(x) = γ ⊙ tanh(λ / (κ·s_x + |μ_x|) · x)`. Single variance reduction per block; lightweight downstream norms.
- **Why here:** RMSNorm has sequential reductions that bottleneck kernels. BHyT is 15.8% faster, prevents depth-wise variance explosion.
- **Cost:** Low drop-in.
- **Risks:** Activation distribution shift requires GPTQ recalibration.
- **Pairs with:** Progressive Recurrence.

#### C.2 Derf (claim: arXiv:2512.10938, Chen et al. 2025)
- **Mechanism:** `Derf(x) = erf(αx + s)` replaces both LN and activation.
- **Why here:** Subcritical signal propagation at init; better than LN/DyT at 100M scale.
- **Cost:** Low.
- **Risks:** Highly sensitive to α init; divergence on bad scaling.
- **Mutually exclusive with BHyT.**

#### C.3 Progressive Depth Recurrence (claim: PR #1440 ablations, April 2026)
- **Mechanism:** Stage-activate the looped layers; deep loops first, shallow loops later.
- **Why here:** SOTA's abrupt activation at frac=0.35 produces a loss spike; staging removes it.
- **Cost:** Trivial scheduling change.
- **Risks:** Negligible.

#### C.4 EngramLite (claim: PR #1089 / #1440)
- **Mechanism:** Multi-order (bi+tri-gram) hash w/ learned-bias sigmoid gate; adds residual to logits.
- **Why here:** Frees attention/MLP from shallow memorization.
- **Cost:** **High byte cost** (~1.3 MB unquantized for 3072×112 table).
- **Risks:** Bad gate init → noisy trigrams hurt logits.
- **Compliance:** OK (training-time hash, not eval-only n-gram cache).

### Area D — Optimizer and training dynamics

#### D.1 Mousse (claim: arXiv:2603.09697, Zhang et al. 2026)
- **Mechanism:** Diagonal Kronecker-factored (Shampoo-style) preconditioner *before* Newton-Schulz orthogonalization. Anisotropic spectral steepest descent.
- **Why here:** Activation `ZZ^T` is ill-conditioned; Muon's uniform spectral norm wastes updates on distorted directions.
- **Cost:** ~3% per-step wallclock overhead. Modifies existing Muon class.
- **Claimed gain:** Pareto −12% steps to equivalent val loss. At fixed time → deeper minimum.
- **Risks:** 3% per-step cost compounds with kernel launch latency → fewer total steps.

#### D.2 NuMuon (claim: arXiv:2603.03597, Dolatabadi et al. 2026)
- **Mechanism:** Nuclear-norm budget on Muon updates; truncated singular directions; interpolates rank-1↔full-rank.
- **Why here:** Coerces low-rank weights → Brotli compresses better → free bytes for capacity.
- **Cost:** Moderate; SVD truncation in optimizer step.
- **Risks:** Over-truncation hurts representational floor.
- **Pairs with:** CERWU.

#### D.3 AdamHD (claim: arXiv:2511.14721)
- **Mechanism:** Huber regularizer instead of L2 weight decay (quadratic below threshold, linear above).
- **Why here:** Suppresses outlier weights → tighter SDClip thresholds → cleaner int6 quant.
- **Cost:** Low — minor AdamW mod (used for embeddings, control vectors).
- **Risks:** Huber threshold needs tuning.

### Area E — Artifact compression

#### E.1 TCA-TBE / ZipServ (claim: arXiv:2603.17435, Fan et al. ASPLOS 2026)
- **Mechanism:** Fixed-length bitmap encoding (vs variable-length ANS/Huffman); ZipGEMM kernel decompresses directly into Tensor Core registers.
- **Why here:** Brotli currently decompresses CPU→host→GPU; TCA-TBE eliminates the staging.
- **Direct gain:** ~10–15s of eval init time → more TTT epochs.
- **Cost:** **High** — Triton/CUDA kernel inside LZMA-wrapped Python.
- **Risks:** Custom kernel code may exceed compressed size budget.

#### E.2 ANS / FSE (Duda 2013, well-known)
- **Mechanism:** Asymmetric numeral systems; arithmetic-coding compression at Huffman-coding speed.
- **Why here:** Brotli is optimized for repetitive text (LZ77); GPTQ byte-shuffled weights are dense probabilistic data → ANS theoretically wins.
- **Direct gain:** ~0.5 MB extra weight space.
- **Cost:** High — must be vectorized in PyTorch (pure Python ANS = too slow for 600s eval).

### Area F — Score-first TTT theory

> The information-theoretic upper bound of score-first TTT is bounded by the *mutual information* between sequential chunks. If chunk N (Wikipedia article) and chunk N+1 (code snippet) are uncorrelated, naive TTT can be net-negative. → Restricting updates to Q projections (qTTT) or MLP final projection (In-Place TTT) prevents global representation collapse.

### Area H — Wildcards

- **Speedrun-style progressive window warmup** (nanoGPT lineage): start train_seq_len at 256, grow linearly to 2048 by step 2000. Quadratic FLOPs reduction on early steps clears the high-loss landscape faster.
- **MASS (Meta-Adaptation with Self-Synthesis)** — covered in A.4. High-risk, high-cost.

---

## What I'd actually try in 3 days, in order

1. **In-Place TTT** as a `TTT_PARAM_FILTER=mlp_proj_only` variant + a new `TTT_OBJECTIVE=ntp_outer_product` env var. Best risk/reward: keeps train pipeline unchanged, isolates the experiment to eval-time, smaller surface than my current `scales` filter, and Gemini ranks it #1.
2. **qTTT** as `TTT_PARAM_FILTER=q_only` (a new filter selecting only `attn.c_q.weight` per layer) with cached K/V. Run a head-to-head matrix: `{all, scales, scales+embed, q_only, mlp_proj_only}` × `{cross_entropy, ntp_outer}`.
3. **Progressive Depth Recurrence** as `ENABLE_LOOPING_AT` becoming a list/curve (e.g. enable layer-5 loop at 0.20, layer-4 at 0.35). Trivial code change, low risk, claimed −0.001 to −0.002.
4. **Mousse + NuMuon** as a Day-3 stretch (training-time changes, requires retrain × 3 seeds → expensive). Only if 1+2+3 leave us short of −0.005.

Skip CERWU/TCA-TBE/ANS for the 3-day window — too much engineering for uncertain gain. Defer to a post-deadline iteration.

---

## Verbatim Deep Research output

The following is the literal Gemini Deep Research transcript provided by sairamen on 2026-04-27. Preserved as-is for reproducibility; supersedes none of the analysis above.

> **Note:** Deep Research formatting (LaTeX math, embedded markdown links) preserved.

### Parameter Golf: L(N,T) Optimization under Extreme Constraint Artifacts

The "Parameter Golf" challenge represents an extreme frontier of neural scaling optimization, where validation loss is minimized subject to rigid, simultaneous boundaries on parameter count, wallclock training time, and artifact serialization size. The current state-of-the-art baseline stands at 1.0810 bits-per-byte (BPB), utilizing an 11-layer architecture, Partial RoPE, Muon and AdamW optimization, GPTQ with Hessian-aware SDClip, and score-first test-time training (TTT). Recent delta compressions at the top of the leaderboard measure between 0.0006 and 0.003 nats, indicating a highly mature optimization landscape where elementary architectural tweaks have been exhausted.

Surpassing the 1.0810 BPB threshold by ≥0.005 nats under a 600-second training budget, a 600-second evaluation budget, and a 16,000,000 decimal byte serialization limit requires a holistic re-engineering of the optimization geometry, the test-time adaptation objectives, and the artifact compression pipeline. The following analysis synthesizes advancements in matrix-level optimization, test-time adaptation, rate-distortion quantization, and architectural efficiencies published between 2022 and April 2026.

#### Top-10 Prioritized Techniques for Implementation

The techniques below represent the highest-yield interventions for the current 11-layer, 512-dimension architecture, ranked by the product of their expected BPB reduction, probability of successful transfer to this highly constrained regime, and inverse implementation cost.

The analysis indicates that the most substantial gains will not come from adding parameters, but rather from mathematically rectifying the optimization geometry (Mousse), aligning test-time updates with autoregressive goals (In-Place TTT), and recovering wallclock time through hardware-aware artifact loading (TCA-TBE).

**In-Place Test-Time Training (In-Place TTT)** — Replaces the generic reconstruction target of standard TTT with a Next-Token-Prediction aligned objective applied exclusively to the MLP projection matrices. By processing chunks in parallel, it maximizes GPU utilization during evaluation while circumventing the catastrophic forgetting that plagues full-model SGD. Expected to yield significant BPB reduction with minimal byte overhead.

**Mousse Optimizer** — Acts as a drop-in replacement for the standard Muon optimizer. Mousse applies Shampoo-style Kronecker-factored curvature preconditioning prior to Newton-Schulz orthogonalization, correcting the inherent right-side anisotropy of language model activations. It consistently reduces required training steps by approximately 12%, effectively elongating the 600-second training budget.

**Query-Only Test-Time Training (qTTT)** — A compute-frugal adaptation strategy that performs a single prefill to cache all Keys and Values, subsequently restricting gradient updates entirely to the Query projections. This slashes the FLOP requirements of TTT, permitting substantially more adaptation epochs within the 600-second evaluation window while protecting the delicate int6 GPTQ weights from corruption.

**CERWU (Compression with Entropy-Regularized Weight Updates)** — A rate-distortion optimal quantization framework that co-optimizes the quantization grid and entropy coding via an Optimal Brain Surgeon approximation. By mathematically minimizing the downstream brotli bit-rate rather than just weight-space error, CERWU can free up substantial bytes within the 16MB budget, allowing for architectural expansion (e.g., wider MLPs).

**Bounded Hyperbolic Tangent (BHyT) Normalization** — Replaces RMSNorm with a bounded S-shaped nonlinearity that computes exact variance statistics only once per block. BHyT guarantees stability against depth-wise variance explosion while executing 15.8% faster than standard layer normalizations, returning precious seconds to the training phase for additional optimization steps.

**Progressive Depth Recurrence** — Instead of activating the 3-layer depth recurrence abruptly (which causes a severe gradient shock and loss spike), Progressive Depth Recurrence introduces the looped layers in staggered phases. This eliminates the loss spike, ensuring monotonic convergence and maximizing the utility of the final optimization steps.

**NuMuon (Nuclear-Norm-Constrained Muon)** — Augments the Muon optimizer by imposing a nuclear-norm constraint on the update direction, explicitly forcing the learned weight matrices toward a low-rank structure. This pushes compressibility directly into the optimization phase, dramatically improving the efficacy of the final brotli compression pass.

**EngramLite (Multi-Head Gated N-gram Hash)** — Replaces standard bi-gram hashing with a multi-order scheme utilizing a sigmoid gate to suppress noisy lookups. By offloading strict n-gram memorization to an embedding table, it preserves the parametric capacity of the 11-layer network for deeper semantic representations.

**Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)** — A fixed-length, bitmap-based lossless encoding that replaces CPU-bound entropy codecs. By enabling parallel decompression directly into Tensor Core registers, TCA-TBE can shave up to 15 seconds off the artifact loading time, directly extending the test-time training budget.

**Dynamic erf (Derf) Activation** — Replaces both normalization and standard activation functions with the rescaled Gaussian cumulative distribution function. Proven to induce subcritical signal propagation, Derf improves structural generalization across small language models, acting as an alternative to BHyT.

#### Area A: Test-Time Training and Adaptation

The 600-second evaluation budget paired with the score-first compliance constraint dictates that test-time training must be highly sample-efficient and immune to localized catastrophic forgetting. The current methodology — applying SGD across all 34M parameters — is computationally inefficient and misaligned with the autoregressive objective.

**In-Place Test-Time Training (In-Place TTT)**
- **Citation:** Feng et al., 2026, arXiv:2604.06169.
- **Mechanism:** Bypasses generic self-supervised reconstruction targets by isolating the final projection matrix of MLP blocks as adaptable "fast weights." Utilizes a Next-Token-Prediction (NTP) aligned target, V_LM = E_{x_{t+1}}, updating fast weights via chunk-wise gradient: W_down^(i) = W_down^(i-1) + η · V̂_[i]^T · Z_[i].
- **Why it might work here:** Score-first constraint requires sequential inference. Standard TTT auto-encodes current tokens (fails to improve causal reasoning). In-Place TTT forces MLP to learn local causal text distribution, aligning with BPB metric.
- **Cost:** Low. Modifies eval script. <500 bytes compressed Python. Eliminates full-model backprop.
- **Δ BPB:** −0.004 to −0.008 (4B-param long-context experiments).
- **Risks:** Aggressive lr could destabilize int6 weights without per-chunk annealing.
- **Synergy:** Perfect with GPTQ — isolates updates to specific projections.
- **Compliance:** OK.

**Query-Only TTT (qTTT)**
- **Citation:** Barua et al., 2025, arXiv:2512.13898.
- **Mechanism:** Single prefill caches K and V; subsequent passes freeze K/V and update only Q projections.
- **Why here:** Full-model SGD underutilizes H100. qTTT bypasses recomputation → 2–3× more adaptation epochs in 600s.
- **Cost:** Low.
- **Δ BPB:** −0.003 to −0.010.
- **Risks:** Q-only may lack capacity for deep distribution shift.
- **Compliance:** OK.

**Large Chunk TTT (LaCT)**
- **Citation:** arXiv:2505.23884.
- **Mechanism:** Document-sized chunk memory updates + local window attention; Muon as fast-weight optimizer at inference.
- **Why here:** Lifts eval GPU utilization from <5% to ~70%.
- **Cost:** Moderate.
- **Δ BPB:** −0.002 to −0.008.
- **Risks:** Severe overfitting to current chunk → harms next.
- **Conflicts with qTTT.**
- **Compliance:** OK if score-first preserved.

**MASS — Meta-Adaptation with Self-Synthesis**
- **Citation:** Zweiger et al., 2026, arXiv:2603.03524.
- **Mechanism:** Bilevel meta-learning of TTT init + hyperparams during pretraining.
- **Why here:** Meta-trained model converges TTT in 1 epoch instead of 3.
- **Cost:** **High.** Bilevel inner/outer in 600s budget.
- **Δ BPB:** −0.002.
- **Risks:** ~50% reduction in primary gradient steps → likely net loss.
- **Competes for the 600s budget.**

#### Area B: Quantization at int6 and below

SOTA uses GPTQ + Hessian-aware SDClip at int6 / int8 embeddings. Frontier shifts from heuristic post-training clipping to mathematically rigorous rate-distortion bounds.

**CERWU**
- **Citation:** Conzelmann & Bamler, 2025, arXiv:2505.18758.
- **Mechanism:** Quantization as explicit rate-distortion optimization: D(W,Ŵ) − λ log_2 P(Ŵ). Co-optimizes grid and weight updates via OBS.
- **Why here:** GPTQ minimizes output distortion only, ignoring compressibility. CERWU explicitly trains for low-entropy weights → better Brotli.
- **Cost:** Mod-High; offline, outside 600s.
- **Δ BPB:** −0.003 to −0.008. Frees 1–3 MB → reinvest in wider MLP / deeper.
- **Pairs with NuMuon.**
- **Compliance:** OK.

**AQLM**
- **Citation:** Egiazarian et al., 2024, arXiv:2401.06118.
- **Mechanism:** Multi-codebook residual vector quantization at the layer-output level.
- **Why here:** Embeddings stuck at int8; AQLM compresses to ~3 bits/param at higher fidelity than int4 scalar.
- **Cost:** High (custom RVQ decoder kernel in LZMA-wrapped Python).
- **Δ BPB:** −0.003.
- **Risks:** Decoder bytes may exceed weight savings on 8K vocab.
- **Synergistic with CERWU (different surfaces).**
- **Compliance:** OK.

**OptRot / ROSAQ**
- **Citation:** Yoon et al., 2024/2025.
- **Mechanism:** Orthogonal rotation (Hadamard / PCA) of weights+activations pre-quantization.
- **Why here:** int6 outlier-sensitive; rotation enables tighter SDClip.
- **Cost:** Low; rotations fuse into adjacent linears (zero param overhead).
- **Δ BPB:** −0.001 to −0.003.
- **Risks:** May interfere with In-Place TTT geometric assumptions.
- **Compliance:** OK.

#### Area C: Architectural micro-changes

Small-scale (11L × 512d) operates in distinct dynamical regime; micro-changes target subcritical signal propagation.

**BHyT Normalization**
- **Citation:** Byun et al., 2026, arXiv:2601.09719.
- **Mechanism:** BHyT(x) = γ ⊙ tanh(λ / (κ·s_x + |μ_x|) · x). Single variance reduction per block.
- **Why here:** RMSNorm reductions bottleneck kernels. BHyT is 15.8% faster, prevents depth-wise variance explosion.
- **Cost:** Low drop-in.
- **Δ BPB:** −0.004.
- **Risks:** Activation distribution shift → GPTQ recalibration needed.
- **Pairs with Progressive Recurrence.**

**Dynamic erf (Derf)**
- **Citation:** Chen et al., 2025, arXiv:2512.10938.
- **Mechanism:** Derf(x) = erf(αx + s) replaces both normalizations and activations.
- **Why here:** Subcritical signal propagation at init; better than LN/DyT at 100M scale.
- **Cost:** Low.
- **Δ BPB:** −0.003.
- **Risks:** α init sensitive.
- **Mutually exclusive with BHyT.**

**Progressive Depth Recurrence**
- **Citation:** PR #1440 ablations, April 2026.
- **Mechanism:** Stage-activate looped layers (deep first, shallow later).
- **Why here:** SOTA's abrupt activation at frac=0.35 produces gradient shock.
- **Cost:** Trivial scheduling change.
- **Δ BPB:** −0.001 to −0.002.
- **Risks:** Negligible.

**EngramLite**
- **Citation:** PR #1089 / #1440.
- **Mechanism:** Multi-order (bi+tri-gram) hash w/ learned-bias sigmoid gate; residual to logits.
- **Why here:** Frees attention/MLP from shallow memorization.
- **Cost:** **High byte cost** (~1.3 MB unquantized for 3072×112 table).
- **Δ BPB:** −0.003.
- **Risks:** Bad gate init → noisy trigrams hurt.
- **Compliance:** OK (training-time hash).

#### Area D: Optimizer + training dynamics

**Mousse**
- **Citation:** Zhang et al., March 2026, arXiv:2603.09697.
- **Mechanism:** Diagonal Kronecker-factored Shampoo-style preconditioner *before* Newton-Schulz. Anisotropic spectral steepest descent.
- **Why here:** Activation ZZ^T is ill-conditioned. Muon's uniform spectral norm wastes updates on distorted directions.
- **Cost:** Low code mod; ~3% per-step wallclock overhead.
- **Δ BPB:** −0.006 to −0.008. Pareto −12% steps to equivalent val loss.
- **Risks:** 3% overhead compounds → fewer total steps.
- **Direct Muon replacement.**

**NuMuon**
- **Citation:** Dolatabadi et al., March 2026, arXiv:2603.03597.
- **Mechanism:** Nuclear-norm budget on Muon updates; truncated singular directions.
- **Why here:** Coerces low-rank weights → Brotli wins → free bytes for capacity.
- **Cost:** Moderate; SVD truncation in optimizer step.
- **Δ BPB:** −0.002 to −0.006.
- **Risks:** Over-truncation hurts representational floor.
- **Pairs with CERWU.**

**AdamHD (Huber Decay)**
- **Citation:** arXiv:2511.14721.
- **Mechanism:** Huber regularizer instead of L2 weight decay (quadratic below threshold, linear above).
- **Why here:** Suppresses outlier weights → tighter SDClip thresholds.
- **Cost:** Low — minor AdamW mod.
- **Δ BPB:** −0.002 to −0.005.
- **Risks:** Threshold needs tuning.

#### Area E: Artifact compression

**TCA-TBE / ZipServ**
- **Citation:** Fan et al., ASPLOS March 2026, arXiv:2603.17435.
- **Mechanism:** Fixed-length bitmap encoding (vs variable-length ANS/Huffman); ZipGEMM kernel decompresses directly into Tensor Core registers.
- **Why here:** Brotli decompresses CPU→host→GPU; TCA-TBE eliminates staging.
- **Direct gain:** ~10–15s eval init → more TTT epochs.
- **Cost:** **High.** Triton/CUDA kernel inside LZMA-wrapped Python.
- **Δ BPB:** −0.001 to −0.002 (indirect).
- **Risks:** Custom kernel may exceed compressed size budget.

**ANS / FSE**
- **Citation:** Duda 2013.
- **Mechanism:** Asymmetric numeral systems — arithmetic-coding compression at Huffman speed.
- **Why here:** Brotli optimized for repetitive text; GPTQ byte-shuffled weights are dense probabilistic data → ANS theoretically wins.
- **Direct gain:** ~0.5 MB extra weight space.
- **Cost:** High — must vectorize in PyTorch (pure Python → too slow for 600s).

#### Area F: Score-first TTT theory

The information-theoretic upper bound of score-first TTT is bounded by mutual information between sequential chunks. If chunk N (Wikipedia article) and N+1 (code snippet) are uncorrelated, naive TTT can be net-negative. → Restricting updates to Q projections (qTTT) or MLP final projection (In-Place TTT) prevents global representation collapse.

#### Area G: Combinations

| T1 | T2 | Synergy | Combined Δ |
|----|----|---------|-----------|
| Mousse | NuMuon | aligned-geometry descent + low-rank weights | −0.005 to −0.009 |
| In-Place TTT | qTTT | MLP fast-weights × Q-only updates | −0.006 to −0.010 |
| Progressive Recur | BHyT | bounded variance prevents shock | −0.004 to −0.007 |
| CERWU | ANS | rate-distortion grid + entropy coder → Shannon limit | −0.005 to −0.008 |

#### Area H: Wildcards

- **MASS** — bilevel meta-learning of TTT during pretrain. High-cost, high-risk.
- **Speedrun progressive window warmup** — start train_seq_len at 256, grow to 2048 by step 2000. Quadratic FLOP reduction early → faster escape from high-loss landscape.

#### Techniques to Drop

- **AdEMAMix** — outperforms only at small-batch / noise-heavy. Loses to Muon/Mousse at large-batch pretraining (8×H100, 350M tokens).
- **TTT-Linear / TTT-MLP (Stanford)** — requires replacing FA3. Architectural rewrite.
- **Attention sinks / register tokens** — for 100K+ context; useless at seq_len=2048.
- **xIELU activation** — no fast kernels; Python fallback blows 600s budget.
- **Test-time LoRA (post-quant)** — fp matrices at eval time → memory overhead, slow.

#### The Single Most Novel Finding

**In-Place TTT with NTP Alignment (April 2026, arXiv:2604.06169).** SOTA's score-first TTT uses generic SGD cross-entropy over the entire network — poor GPU utilization, risks GPTQ corruption, misaligned with causal objective. In-Place TTT shows architectural rewrites (Stanford TTT-Linear) are unnecessary. Isolating MLP final projection as fast weights + NTP target (V_LM = E[x_{t+1}]) forces adaptation toward causal text distributions. Update rule W^(i) = W^(i-1) + η · V̂_[i]^T · Z_[i] is matrix-vector → chunk-parallel. Solves all primary bottlenecks: minimizes eval FLOPs, protects attention quant, avoids catastrophic forgetting, mathematically aligns with BPB. Highest probabilistic path to breaking 1.0810.
