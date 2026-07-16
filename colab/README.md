# SigLIP support + SAE interpretability for vision encoders

This directory holds the training/eval work that accompanies the SigLIP support
added to ViT-Prisma. It is a BlueDot AI Safety course project on the mechanistic
interpretability of frozen vision encoders, framed around silent perception
failures in high-stakes systems (autonomous driving, robotics).

## Motivation

Vision-language systems inherit a frozen, pretrained encoder (CLIP, SigLIP) as
their perception front end. When that encoder fails silently on an unusual input,
the whole system acts on a wrong model of the world with no warning. Sparse
autoencoders (SAEs) make an encoder's internal features human-readable, so they
can be audited before deployment rather than after harm.

Two gaps motivated this work:

1. **Tooling.** ViT-Prisma could not load SigLIP at all, so it was outside the
   interpretability stack. This project adds SigLIP1/SigLIP2 support (loader,
   config registry, weight conversion) for `google/siglip-base-patch16-224` and
   `google/siglip2-base-patch16-224`.
2. **A clean question.** CLIP, SigLIP, and a supervised ImageNet ViT all share an
   identical ViT-B/16 backbone (12 layers). With architecture held fixed, the
   *training objective* is the only variable. Does the objective change how
   interpretable — how sparsely reconstructable — the internal features are?

## What's here

**Model support (in `src/vit_prisma/models/`):** `model_config_registry.py`,
`model_loader.py`, `weight_conversion.py` — SigLIP1/2 load and hook cleanly at all
12 layers (`hook_resid_post`, `hook_mlp_out`).

**Demos (`demos/`):**
- `SigLIP_Demo.ipynb` — loads SigLIP1/2, verifies activation caching at every layer.
- `Vision_LM_Logit_Lens.ipynb` — patch-level text-similarity logit lens comparing
  CLIP vs SigLIP1 vs SigLIP2 at layers 0/4/8/11 on a cat+dog image.

**SAE pipeline (`colab/`):**
- `SAE_Training.ipynb` (SigLIP), `SAE_Training_CLIP.ipynb`, `SAE_Training_ViT.ipynb`
  — matched top-k SAEs on frozen `hook_resid_post`, all 12 layers, expansion
  factor 4 (d_sae = 3072), ImageNet-val (50K), Colab A100. Resume-safe checkpointing.
- `SAE_Eval_Comparison.ipynb`, `SAE_Eval_Wandb.ipynb`, `Plot_SAE_Metrics.ipynb`
  — pull metrics from wandb, health-check, and plot cross-model comparisons.
- `SAE_LogitLens_Comparison.ipynb` — runs the logit lens on SAE *reconstructions*
  (encode → decode) vs raw activations, as a faithfulness probe.
- `Download_Dataset.ipynb` — ImageNet-val streaming helper.

> Trained checkpoints, CSVs, and plots live under `colab/results/` and Drive, which
> are gitignored. The notebooks regenerate them. Notebooks use a bare
> `wandb.login()`; **no credentials are embedded.**

## Method

- Sparsity via **top-k** activation: `k` directly sets L0 (active features/patch).
  relu+L1 collapses on these activations (L0 < 1.5 regardless of the L1 coefficient
  across a 10x range) — a documented negative result. Chosen: **k = 128**.
- **Explained variance (EV)** is the metric of record; it is comparable across
  layers, whereas raw MSE is not (residual-stream norms grow with depth).

## Key results

1. **relu+L1 collapses; top-k is the fix.** k directly controls L0, with zero dead
   features at every k tried (32/64/128).
2. **EV rises log-linearly with k** (~+0.08 per doubling) and **declines with depth**
   for all three models — a depth/architecture effect, not objective-specific.
3. **The objective sets feature compressibility on the same backbone:**
   - *Supervised ViT* — highest EV at layer 0 (0.94), steepest decline.
   - *CLIP (contrastive)* — lowest MSE at every layer; a mid-network EV bump.
     Contrastive pooling yields a low-rank late representation.
   - *SigLIP (sigmoid)* — smoothest decline, highest EV at the final layer. Sigmoid
     loss preserves distributed per-patch structure.
4. **SAE-reconstruction faithfulness mirrors this** (per-patch logit-lens agreement,
   original vs SAE): CLIP rises monotonically with depth (75 → 95%) as it pools
   toward one dominant concept; SigLIP is U-shaped and lower (78 → 64 → 65 → 75%)
   because its mid layers keep higher-rank spatial diversity that resists sparse
   compression. Same objective effect, seen qualitatively.

## Conclusion

Holding architecture fixed, the training objective measurably shapes feature
geometry. CLIP compresses cleanly into few sparse features; SigLIP resists sparse
compression exactly where its distributed per-patch structure lives; supervised ViT
is most compressible early and least deep. So *"which objective yields more
interpretable features"* has a real, measurable, depth-dependent answer.

This is a **50K-image proof of concept**: the *contrast between models* is the robust
signal, not the absolute EV/agreement numbers, which need full-ImageNet scale to
reach analysis grade. The reusable outputs — SigLIP support in ViT-Prisma and the
matched SAEs — stand on their own.

## Limitations

- 50K images / 3 passes; deep-layer EV (~0.68–0.70) is below the ~0.9 that
  full-scale SAEs reach, so deep-layer claims are proof-of-concept.
- Single seed; the logit-lens agreement numbers come from one cat/dog image.
- EV measures reconstruction, not monosemanticity.
- The matched SigLIP-vs-CLIP-vs-ViT SAE framing is plausibly novel but not yet
  verified against the literature; treat as preliminary.

## Would-be next steps (out of scope without compute)

Capacity/k sweep to separate intrinsic structure from SAE undertraining; scale to
full ImageNet (1.28M); feature interpretation via max-activating images; a driving
-dataset robustness probe (do safety-relevant features survive fog/night/blur?);
add SigLIP2 and `hook_mlp_out` to the comparison.
