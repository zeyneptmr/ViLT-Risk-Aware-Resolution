# Risk-Aware Resolution Allocation for ViLT

This project explores whether image resolution can be selected based on the question to reduce visual patch usage while keeping most of the Visual Question Answering (VQA) performance.

I use the pretrained [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) checkpoint without changing its weights. Images are processed at short-edge targets of either 256 or 384 pixels.

**Reference paper:** Kim et al., [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://proceedings.mlr.press/v139/kim21k.html), ICML 2021.

## Experiment

- **Dataset:** VQAv2 validation questions and annotations with COCO val2014 images.
- **Data split:** 1,000 development and 2,000 final image-question pairs in the LARGE experiment. Only one question is selected per image, and the two sets do not share images.
- **Seed:** 5110 for data selection.
- **Resolutions:** 256 and 384 pixels, with a fixed 32 × 32 patch size.
- **Methods:** Always-384, Always-256, Matched Random, yes/no heuristic, Category-Sensitivity, and Risk-Aware.
- **Budgets:** 60%, 75%, and 90% of the Always-384 visual patch total.
- **Evaluation:** VQAv2 scores, visual patch counts, 95% confidence intervals from 2,000 paired bootstrap samples, and inference timing on 24 questions using an NVIDIA L4 GPU.

The resolution selection method uses development-set results to estimate which questions may benefit from higher resolution. These estimates are then kept fixed for the final evaluation.

Risk-Aware uses question categories and additional text cues to decide which images receive higher resolution. The method distributes a limited patch budget across a known group of questions.

No model retraining is needed. After choosing the resolution, ViLT runs once for each image-question pair. Both resolutions were evaluated separately to compare the policies in the experiments.

## Results

The LARGE experiment produced the following VQA scores:

| Method | 60% budget | 75% budget | 90% budget |
|---|---:|---:|---:|
| Matched Random (mean) | 77.26% | 78.25% | 79.16% |
| Category-Sensitivity | 77.65% | 79.06% | 79.68% |
| Risk-Aware B + D | 77.35% | 78.90% | 79.60% |

Always-384 achieved **79.81%** on the final validation subset.

At the 75% patch budget, Risk-Aware achieved **78.90%**. It used about 25% fewer visual patches while losing 0.91 percentage points compared with Always-384.

Risk-Aware performed better than Matched Random at the 75% and 90% budgets. However, Category-Sensitivity gave slightly higher scores at all three budgets.

The separate latency measurement did not show a meaningful improvement in inference time.

These results are based on a validation-derived subset and should not be directly compared with the original ViLT paper's test-dev results.

## How to Run

### 1. Open the notebook

Open [`ViLT_Risk_Aware_Resolution.ipynb`](ViLT_Risk_Aware_Resolution.ipynb) in Google Colab.

Select a GPU runtime. The reported LARGE experiment was run on an NVIDIA L4.

### 2. Set up Google Drive

Run the notebook cells in order.

The notebook mounts your own Google Drive and uses the following directory for downloaded data and saved results:

`MyDrive/YZM5110_ViLT_Risk_Aware`

You can change this path in the notebook if needed.

### 3. Choose the experiment size

In the configuration cell, set `RUN_SIZE` to one of the following:

- `SMALL` — a smaller run for testing the workflow.
- `MEDIUM` — an intermediate experiment.
- `LARGE` — the configuration used for the reported final results.

### 4. Install dependencies

The notebook includes its package installation and configuration steps. It requires Python, PyTorch, Hugging Face Transformers, and other packages imported by the notebook.

The recorded environment and configuration are available in [`results/protocol/configuration.json`](results/protocol/configuration.json).

### 5. Download the required data

The notebook uses the official VQAv2 validation questions and annotations, COCO val2014 images, and the pretrained ViLT checkpoint.

**Data sources:**

- [VQAv2](https://visualqa.org/download.html)
- [COCO](https://cocodataset.org/)
- [ViLT checkpoint](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

The notebook downloads or loads the required files when running. An internet connection and sufficient Google Drive storage are needed.

The full datasets, COCO images, downloaded model weights, and large prediction caches are not included in this repository.

### 6. Check the outputs

The notebook saves experimental settings, selected data splits, predictions, resolution decisions, metrics, and figures.

The final results can be inspected in the `results/` directory.

The reported experiment used checkpoint revision:

`d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08`

## Repository Structure

```text
ViLT-Risk-Aware-Resolution/
│
├── README.md
├── ViLT_Risk_Aware_Resolution.ipynb
│
└── results/
    ├── metrics/
    │   ├── final_method_comparison.csv
    │   ├── matched_budget_effects.json
    │   └── ...
    │
    ├── figures/
    │   ├── accuracy_vs_patch_budget.png
    │   └── accuracy_vs_measured_latency.png
    │
    └── protocol/
        ├── configuration.json
        ├── checkpoint.json
        ├── manifest.csv
        ├── frozen_policy.json
        └── ...
```

The `metrics` folder contains evaluation results and comparisons. The `figures` folder contains the saved plots. The `protocol` folder contains the experiment settings, data split, allocation decisions, and other information needed to examine the run.

The notebook contains the code used to generate the experimental results and figures.

## Evaluation Scope and Limitations

The development and final sets are separated at the image level. However, the exact validation images used during fine-tuning of the released checkpoint could not be verified.

For this reason, the results should be interpreted as comparisons between resolution selection methods on the same checkpoint, rather than as performance on completely unseen images.

The method also assumes that the group of questions is known before the total patch budget is allocated. It is not a fully independent resolution decision for each incoming question.

Finally, reducing visual patches did not lead to a measurable speed improvement in the small latency experiment. The main resource saving reported here is the reduction in visual patch usage.

## Reproducibility

The notebook uses deterministic data selection and records the settings needed to examine the experiments. Saved results include evaluation metrics, allocation decisions, data splits, and timing measurements.

The reported results belong to the LARGE experiment. A smaller configuration can be used to test the workflow before running the full experiment.

The submitted version of the project can be identified by its Git commit or tag.
