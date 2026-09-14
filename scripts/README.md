# Scripts

Included scripts preserve the Paper 1 downstream statistical and modeling logic.

- `statistics/icc_pairwise_axisA.py`: pairwise ICC(A,1)/ICC(C,1) for Axis A.
- `statistics/estadistico.py`: robustness, SRAD, Friedman/Wilcoxon and discrimination summaries.
- `feature_selection/ensemble.py`: auxiliary cross-fold recurrence/stability feature selection.
- `modeling/modeling.py`: leakage-free LR/XGB/MLP internal modeling.
- `validation/external_val.py`: frozen external validation using saved bundles.
- `validation/bootstrap_auc_ci.py`: patient-level bootstrap AUC CIs and paired delta-AUC.
- `figures/plot_modeling_results.py`: modeling summary figure helper.
- `binwidth.py`: bin-width summary helper.

Some scripts retain disabled/general code paths for ComBat or density parsing because they are part of the original implementation. The release commands and packaged inputs are restricted to Axis A B/OFF N/OFF C/OFF.
