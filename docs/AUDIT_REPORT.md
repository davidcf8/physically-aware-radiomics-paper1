# Audit Report

## Included Analysis Scope

This release includes only final Paper 1 Axis A analyses: NR, RS, VS and FK with B=OFF, N=OFF, C=OFF. No density, ComBat, full factorial/Paper 4, exploratory, cache or raw imaging outputs were intentionally included.

The release is suitable for freezing as `v1.0.0-paper1`. Several included scripts remain general-purpose and retain dormant code paths for additional axes. This is intentional: those scripts were the authoritative Paper 1 framework, and disabling code paths would change the software snapshot. The release boundary is therefore enforced by packaged data, packaged reference outputs and documented commands, all of which are Axis A-only.

## Script Classification

| File | Decision | Reason |
|---|---|---|
| `modeling.py` | REQUIRED | Final train-only common-neutral, per-arm, full-feature, LR, XGB and MLP modeling logic. |
| `external_val.py` | REQUIRED | Frozen external validation from saved bundles and external derived feature CSVs. |
| `ensemble.py` | REQUIRED | Auxiliary five-fold recurrence/stability feature-selection analysis. |
| `bootstrap_auc_ci.py` | REQUIRED | Final patient-level bootstrap AUC CI and paired delta-AUC analysis. Copied from final bootstrap output directory. |
| `icc_pairwise_axisA.py` | REQUIRED | Axis A pairwise ICC(A,1)/ICC(C,1). |
| `estadistico.py` | REQUIRED | Robustness summaries, SRAD, Friedman/Wilcoxon and discrimination exports. |
| `binwidth.py` | SUPPORTING | Bin-width summaries. |
| `plot_modeling_results.py` | SUPPORTING | Modeling summary figures/tables. |
| `run_modeling_axisA_sequential_nuevos.sh` | SUPPORTING | Example Axis A orchestration; paths sanitized. |
| `run_external_val_nuevos.sh` | SUPPORTING | Example external validation orchestration; paths sanitized. |
| `ICC_paper4.py` | EXCLUDED | Paper 4/full A/B/C factorial ICC script. |
| `run_abc_density_combat_experiment.py` | EXCLUDED | Density/ComBat factorial experiment. |
| `audit_abc_outputs.py` | EXCLUDED | Audit helper for excluded factorial outputs. |
| `external_val_full_arms.py` | EXCLUDED | Full-arm/factorial external validation path, not final Paper 1 Axis A-only. |
| Paper4/full/resume shell runners | EXCLUDED | Paper 4, density/ComBat or local tmux orchestration. |
| `__pycache__` | EXCLUDED | Cache files. |

## Data Selection

Feature CSVs were selected from final `source_inventory.csv` files for the authoritative Axis A runs and external validation ready folders. The release copies were privacy-reduced to pseudonymized IDs, outcome/split columns and `original_`/`gradient_` radiomic features.

## Code Cleaning

Absolute local paths in copied scripts/reference text were replaced with placeholders or release-relative paths. `bootstrap_auc_ci.py` was edited in the release only so that it reads packaged reference outputs and writes to `results/reproduced/bootstrap_auc_ci`.

## Strict Recurrence Note

The saved strict recurrence outputs use `min_votes=5` and `stable_frac=0.6`, i.e. at least 5 of 6 selectors within fold and at least 3 of 5 folds. The relaxed outputs use `min_votes=4` and `stable_frac=0.4`.
