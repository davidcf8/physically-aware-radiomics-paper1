# Physically Aware Radiomics Without Intensity Interpolation: Paper 1 Reproducibility Package

This repository contains the downstream reproducibility package for Paper 1 of **"Physically Aware Radiomics Without Intensity Interpolation: Disentangling Voxel Geometry and Signal Modification in CT and MRI"**.

## Scope

This repository reproduces the Paper 1 downstream analyses from derived radiomic feature tables. It does **not** contain raw CT/MRI images, segmentation masks, or the modified PyRadiomics voxel-spacing-aware implementation. The voxel-spacing-aware PyRadiomics implementation belongs to the separate technical Paper 2 repository and should be cited/retrieved there.

The reproducibility snapshot associated with Paper 1 is frozen as Git tag `v1.0.0-paper1`.

The data and reference outputs distributed in this release correspond exclusively to Axis A:

- `A = NR, RS, VS, FK`
- `B = OFF`
- `N = OFF`
- `C = OFF`
- no ComBat analysis
- no density-correction or density-harmonization analysis

Some included scripts are the original general-purpose Paper 1 analysis framework and still contain code paths for additional experimental axes. Those general code paths are intentionally preserved to avoid altering the authoritative implementation. They are not used by the Paper 1 reproduction commands, and no data or reference outputs from those additional axes are distributed here.

## Cohorts

Internal predictive cohorts:

- CT: 673 patients total; 538 train and 135 held-out test.
- MRI: 208 patients total; 163 train and 45 held-out test.

External validation cohorts:

- CT: 46 patients.
- MRI: 63 patients.

The CSVs use pseudonymized `patient_id` values. Direct image/mask paths, DICOM dates, age, sex, scanner metadata, and redundant original-ID columns were removed from the release copies.

## Repository Structure

```text
data/derived_features/        Final derived radiomic feature tables used in the Axis A analyses.
scripts/statistics/           ICC, SRAD, Friedman/Wilcoxon, biological discrimination scripts.
scripts/feature_selection/    Auxiliary recurrence/stability feature-selection script.
scripts/modeling/             Internal LR/XGB/MLP modeling pipeline and example commands.
scripts/validation/           External frozen validation and bootstrap AUC CI scripts.
scripts/figures/              Modeling figure helper.
configs/                      Axis A release-local path map.
results/reference_outputs/    Machine-readable final outputs used to verify the published analysis.
docs/                         Audit report, analysis map, data dictionary and reproducibility notes.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `xgboost`, and `lightgbm` are required for full model regeneration. The reference-output checks do not retrain models.

## Reproducibility Quick Check

From the repository root:

```bash
python scripts/validation/reproducibility_check.py
```

This validates cohort sizes, feature counts, per-arm predictive signature sizes, recurrence counts, bootstrap AUC tables, and paired delta-AUC counts against the packaged authoritative reference outputs.

## Example Commands

Pairwise ICC for Axis A:

```bash
python scripts/statistics/icc_pairwise_axisA.py \
  --root_dir data/derived_features/CT/internal \
  --modality CT \
  --B no_density \
  --C OFF \
  --out_dir results/reproduced/statistics/CT_pairwise_icc
```

Auxiliary relaxed recurrence:

```bash
python scripts/feature_selection/ensemble.py \
  --data_dir data/derived_features/CT/internal \
  --out_dir results/reproduced/feature_selection/CT_relaxed \
  --axisA_only --min_votes 4 --stable_frac 0.4 --random_state 42
```

Internal common-neutral modeling with MLP:

```bash
python scripts/modeling/modeling.py \
  --ready_dir data/derived_features/CT/internal \
  --outdir results/reproduced/modeling \
  --analysis_label CT_axisA_common_neutral_mlp \
  --axisA_only --modality CT --feature_mode COMMON_NEUTRAL \
  --drop_shape --include_mlp --seed 1337
```

Model binaries are not included in this release. Therefore, external validation cannot be regenerated from trained model objects using this repository alone. The final patient-level external predictions and machine-readable summary outputs used in the study are provided under `results/reference_outputs/external_validation/`.

Bootstrap CIs from packaged prediction outputs:

```bash
python scripts/validation/bootstrap_auc_ci.py
```

## Expected Key Outputs

- Common-neutral internal and external AUC CIs: `results/reference_outputs/bootstrap/axisA_auc_bootstrap_ci.csv`
- Paired delta-AUC results: `results/reference_outputs/bootstrap/axisA_auc_paired_differences.csv`
- Final model summaries: `results/reference_outputs/modeling/*/results_models.csv`
- External validation summaries: `results/reference_outputs/external_validation/*/external_validation_summary_all.csv`
- Final selected feature lists: `results/reference_outputs/modeling/*/selected_features/`
- Auxiliary recurrence summaries: `results/reference_outputs/feature_selection/*/master_arm_summary.csv`

## Random Seeds

- Internal modeling and bootstrap: `1337` where applicable.
- Auxiliary cross-fold recurrence analysis: `42`.

## Data Availability and Privacy

Only derived radiomic feature tables and derived results are included. Raw medical images and masks are not included. Release CSVs have been reduced to pseudonymized identifiers, outcomes/splits, and radiomic features.

## Citation

If you use the software, derived feature tables, or reference outputs distributed in this repository, please cite both the associated Paper 1 publication and the archived software release.

Paper 1 citation: to be added upon publication.

Software release citation: to be added after archival DOI assignment.

The voxel-spacing-aware PyRadiomics implementation is maintained separately as part of the companion technical study (Paper 2) and is not distributed in this repository.
