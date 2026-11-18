# Risk-Aware Resolution Allocation for ViLT

This project studies whether image resolution can be selected using question features to reduce visual patch usage while preserving Visual Question Answering (VQA) performance. It uses the published, frozen [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) checkpoint and compares 384-pixel and 256-pixel shortest-edge image processing. No model fine-tuning is performed.

**Reference paper:** Kim et al., [*ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision*](https://proceedings.mlr.press/v139/kim21k.html), ICML 2021.

## Experiment

- **Dataset:** VQAv2 validation questions and annotations, with COCO val2014 images.
- **Split:** 1,000 development and 2,000 final question–image pairs in the LARGE configuration; one question per image and no overlap of images between the two locally defined groups.
- **Seed:** 5110 for the deterministic selection protocol.
- **Resolutions:** 384 and 256 pixels (shortest-edge targets), using the same 32 × 32 ViLT patch size.
- **Methods:** always 384, always 256, matched random allocation (20 seeds), a yes/no heuristic, category-based allocation, and Risk-Aware allocation using question categories and text cues.
- **Budgets:** 60%, 75%, and 90% of the all-384 visual patch count.
- **Measurements:** VQAv2 answer scores, visual patch counts, paired bootstrap uncertainty (2,000 resamples), and a separate live-forward latency measurement using 24 questions on an NVIDIA L4 GPU.

The resolution policy is estimated using development-set scores and then evaluated without retuning on the final subset. Its decisions allocate a fixed patch budget across a known set of questions. The frozen ViLT model performs one inference at the selected resolution during policy execution; the experiments compute both resolutions offline to evaluate alternative policies.

## Selected LARGE results

| Method | 60% budget | 75% budget | 90% budget |
|---|---:|---:|---:|
| Matched random (mean) | 77.26 | 78.25 | 79.16 |
| Category-Sensitivity | 77.65 | 79.06 | 79.68 |
| Risk-Aware B + D | 77.35 | 78.90 | 79.60 |

The all-384 baseline obtained **79.81%** on the local final validation-derived subset. At a 75% patch budget, Risk-Aware achieved **78.90%**. These scores are **not directly comparable** to ViLT's published test-dev results. The 24-question latency check did not demonstrate a meaningful end-to-end speedup.

## Run in Google Colab

1. Open `notebooks/ViLT_Risk_Aware_Resolution.ipynb` in Google Colab and select a **GPU** runtime (the recorded LARGE run used an NVIDIA L4).
2. Execute cells in order. The notebook mounts **your own** Google Drive and stores downloaded inputs and computed outputs under `MyDrive/YZM5110_ViLT_Risk_Aware`.
3. In the configuration code cell, choose `RUN_SIZE = 'SMALL'`, `'MEDIUM'`, or `'LARGE'` (the reported final results use `'LARGE'`). Each size uses the same deterministic master split protocol.
4. The notebook installs its explicitly pinned Transformers-related packages if needed and requires a recent PyTorch installation (version >= 2.6). Exact package versions and hardware information are recorded in `results/protocol/configuration.json`.
5. Official VQAv2 questions/annotations, required COCO images, and the published ViLT checkpoint are downloaded or loaded as needed. The first complete execution needs internet access and adequate Google Drive storage.
6. Inspect saved CSV/JSON results and PNG charts. The notebook contains the code that generates these outputs.

**Model checkpoint:** `dandelin/vilt-b32-finetuned-vqa`, pinned revision `d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08`.

**Dataset sources:** [VQAv2 downloads](https://visualqa.org/download.html) · [COCO](https://cocodataset.org/). The full datasets, original COCO JPEGs, downloaded model weights, and prediction caches are **not included** in the repository. Follow each source's usage terms.

## Repository contents

```text
notebooks/
  ViLT_Risk_Aware_Resolution.ipynb
results/
  metrics/       # Final scores, group results, paired cases, latency
  figures/       # Patch-budget and measured-latency plots
  protocol/      # Manifest, frozen policy, allocations, config, provenance
README.md
```

The provided results correspond to the **LARGE** experiment. The `results/protocol` files document the saved cohort, allocations, and run configuration; the notebook regenerates the reported analyses and figures from source data.

## Evaluation scope

The development and final policy-evaluation subsets are image-disjoint from **each other**. However, the original ViLT VQA training protocol can use some VQAv2 validation images, and the exact held-out image list for this released checkpoint has not been verified. Accordingly, the reported numbers are a **within-checkpoint comparison of resolution-allocation policies**, not a claim of generalization to completely unseen checkpoint training inputs.

The reported patch savings are a reduction in **visual input patches**, not a demonstrated equivalent reduction in GPU runtime. The live timing experiment was performed on a limited 24-question subset.

## Reproducibility

The notebook saves the source dataset hashes, the selected model revision, seeded data split, per-question choices, model predictions, aggregate evaluation metrics, bootstrap comparisons, and timing measurements. `configuration.json` records the execution environment for the reported run. To reproduce the exact released version, use the repository commit/tag referenced in the submitted report.
