# Analysis Map

| Manuscript result | Input data | Script/function | Reference output |
|---|---|---|---|
| Bin-width table | extraction summaries / derived feature diagnostics | `scripts/binwidth.py` | `results/reference_outputs/` where available |
| Pairwise ICC(A,1), ICC(C,1) | `data/derived_features/*/internal/merged_homogeneous/*.csv` | `statistics/icc_pairwise_axisA.py`, `compute_pairwise_icc()` | pairwise ICC CSVs generated under `results/reproduced/statistics/` |
| wCV/rMAD/SRAD/Friedman/Wilcoxon/discrimination | same Axis A feature CSVs | `statistics/estadistico.py` | reproduced statistics tables |
| Auxiliary recurrence | Axis A internal feature CSVs | `feature_selection/ensemble.py`, `run_arm_cv_ranking()` | `results/reference_outputs/feature_selection/*` |
| Common-neutral models | Axis A internal feature CSVs | `modeling/modeling.py`, train-only rankings and `build_common_signature_from_rankings()` | `results/reference_outputs/modeling/*_axisA_common_neutral_mlp/results_models.csv` |
| Per-arm models | Axis A internal feature CSVs | `modeling/modeling.py`, `finalize_top_features_from_ranking()` | `results/reference_outputs/modeling/*_axisA_per_arm_ensemble/results_models.csv` |
| Full-feature models | Axis A internal feature CSVs | `modeling/modeling.py`, `get_all_model_features()` | `results/reference_outputs/modeling/*_axisA_full_features/results_models.csv` |
| MLP models | Axis A internal feature CSVs | `modeling/modeling.py`, MLP branch | common-neutral MLP-inclusive summaries |
| Internal bootstrap AUC CIs | held-out prediction CSVs | `validation/bootstrap_auc_ci.py`, `compute_ci_table()` | `results/reference_outputs/bootstrap/axisA_auc_bootstrap_ci.csv` |
| Paired delta-AUC bootstrap | held-out prediction CSVs | `validation/bootstrap_auc_ci.py`, `compute_paired_table()` | `results/reference_outputs/bootstrap/axisA_auc_paired_differences.csv` |
| External validation | external derived feature CSVs plus trained bundles | `validation/external_val.py`, `evaluate_one_bundle()` | `results/reference_outputs/external_validation/*/external_validation_summary_all.csv` |
| Calibration metrics | prediction probabilities | `modeling.py`/`external_val.py`, `evaluate_binary()` and `ece_score()` | model/external result CSVs |
| Figures 1-5 | Mixed: conceptual and generated summaries | `figures/plot_modeling_results.py` where data-driven | conceptual figures are not regenerated from study data here |
| Supplementary modeling figures/tables | reference modeling CSVs | `figures/plot_modeling_results.py` and downstream table formatting | generated under `results/reproduced/` |
