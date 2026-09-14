# Reproducibility Check Report

Date: 2026-09-14

The release check was run from the clean repository with:

```bash
python scripts/validation/reproducibility_check.py
python -m py_compile scripts/statistics/icc_pairwise_axisA.py scripts/statistics/estadistico.py scripts/feature_selection/ensemble.py scripts/modeling/modeling.py scripts/validation/external_val.py scripts/validation/bootstrap_auc_ci.py scripts/validation/reproducibility_check.py scripts/figures/plot_modeling_results.py scripts/binwidth.py
python scripts/validation/bootstrap_auc_ci.py
```

## Results

| Check | Status |
|---|---|
| CT internal cohort n=673 and split 538/135 | PASS |
| MRI internal cohort n=208 and split 163/45 | PASS |
| CT external cohort n=46 | PASS |
| MRI external cohort n=63 | PASS |
| Final CT per-arm predictive feature counts 5/8/12/9 | PASS |
| Final MRI per-arm predictive feature counts 16/13/12/10 | PASS |
| Relaxed recurrence counts CT 7/11/6/7 and MRI 21/26/20/22 | PASS |
| Strict recurrence counts CT 0/1/0/2 and MRI 0/9/1/3 | PASS |
| Bootstrap CI table loads with 48 rows and 10,000 replicates | PASS |
| AUC spot checks for internal/external CT/MRI | PASS |
| Internal paired bootstrap has 24 comparisons, 19 CIs including zero, 5 excluding zero | PASS |
| Python syntax compilation for included scripts | PASS |
| Recomputed bootstrap numeric outputs match reference outputs | PASS |

The recomputed bootstrap CSV differs from the packaged reference CSV only in source-path columns, because the release script writes package-relative paths whereas the original reference CSV recorded the original study tree. Numeric columns and non-path semantic columns match exactly.

## Not Reproduced From Packaged Inputs

External frozen validation from model binaries was not rerun because model bundles are intentionally not distributed in the public release. The final external prediction CSVs and summary CSVs are included as reference outputs for verification.
