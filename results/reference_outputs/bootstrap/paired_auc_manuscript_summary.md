# Paired Bootstrap Delta AUC Summary

Source file: `axisA_auc_paired_differences.csv`. Paired bootstrap resampled the same patient indices for both arms after inner-joining by patient ID and verifying label agreement. Interpretation below is based on whether the percentile 95% CI for delta AUC contains 0; exploratory bootstrap p-values in the source file are not used for manuscript inference.

## CT Internal

| Model | Comparison | AUC arm 1 | AUC arm 2 | Delta AUC (95% CI) | 95% CI contains 0 |
| --- | --- | ---: | ---: | ---: | --- |
| LR | VS - NR | 0.763 | 0.763 | 0.000 (0.000 to 0.000) | Yes |
| LR | RS - NR | 0.730 | 0.763 | -0.033 (-0.072 to 0.004) | Yes |
| LR | FK - NR | 0.763 | 0.763 | 0.000 (0.000 to 0.000) | Yes |
| LR | VS - RS | 0.763 | 0.730 | 0.033 (-0.004 to 0.072) | Yes |
| XGB | VS - NR | 0.763 | 0.763 | 0.000 (0.000 to 0.000) | Yes |
| XGB | RS - NR | 0.711 | 0.763 | -0.052 (-0.119 to 0.017) | Yes |
| XGB | FK - NR | 0.763 | 0.763 | 0.000 (0.000 to 0.000) | Yes |
| XGB | VS - RS | 0.763 | 0.711 | 0.052 (-0.015 to 0.119) | Yes |
| MLP | VS - NR | 0.782 | 0.782 | 0.000 (0.000 to 0.000) | Yes |
| MLP | RS - NR | 0.732 | 0.782 | -0.050 (-0.094 to -0.007) | No |
| MLP | FK - NR | 0.782 | 0.782 | 0.000 (0.000 to 0.000) | Yes |
| MLP | VS - RS | 0.782 | 0.732 | 0.050 (0.007 to 0.093) | No |

## CT External

| Model | Comparison | AUC arm 1 | AUC arm 2 | Delta AUC (95% CI) | 95% CI contains 0 |
| --- | --- | ---: | ---: | ---: | --- |
| LR | VS - NR | 0.616 | 0.616 | 0.000 (0.000 to 0.000) | Yes |
| LR | RS - NR | 0.465 | 0.616 | -0.151 (-0.278 to -0.025) | No |
| LR | FK - NR | 0.616 | 0.616 | 0.000 (0.000 to 0.000) | Yes |
| LR | VS - RS | 0.616 | 0.465 | 0.151 (0.028 to 0.277) | No |
| XGB | VS - NR | 0.590 | 0.590 | 0.000 (0.000 to 0.000) | Yes |
| XGB | RS - NR | 0.284 | 0.590 | -0.305 (-0.524 to -0.080) | No |
| XGB | FK - NR | 0.590 | 0.590 | 0.000 (0.000 to 0.000) | Yes |
| XGB | VS - RS | 0.590 | 0.284 | 0.305 (0.077 to 0.526) | No |
| MLP | VS - NR | 0.648 | 0.648 | 0.000 (0.000 to 0.000) | Yes |
| MLP | RS - NR | 0.527 | 0.648 | -0.121 (-0.246 to 0.000) | Yes |
| MLP | FK - NR | 0.648 | 0.648 | 0.000 (0.000 to 0.000) | Yes |
| MLP | VS - RS | 0.648 | 0.527 | 0.121 (0.000 to 0.246) | Yes |

## MRI Internal

| Model | Comparison | AUC arm 1 | AUC arm 2 | Delta AUC (95% CI) | 95% CI contains 0 |
| --- | --- | ---: | ---: | ---: | --- |
| LR | VS - NR | 0.700 | 0.638 | 0.062 (-0.046 to 0.185) | Yes |
| LR | RS - NR | 0.599 | 0.638 | -0.039 (-0.233 to 0.162) | Yes |
| LR | FK - NR | 0.574 | 0.638 | -0.065 (-0.250 to 0.128) | Yes |
| LR | VS - RS | 0.700 | 0.599 | 0.101 (-0.123 to 0.317) | Yes |
| XGB | VS - NR | 0.380 | 0.558 | -0.177 (-0.343 to -0.017) | No |
| XGB | RS - NR | 0.629 | 0.558 | 0.071 (-0.102 to 0.259) | Yes |
| XGB | FK - NR | 0.521 | 0.558 | -0.037 (-0.212 to 0.130) | Yes |
| XGB | VS - RS | 0.380 | 0.629 | -0.249 (-0.452 to -0.049) | No |
| MLP | VS - NR | 0.482 | 0.703 | -0.221 (-0.429 to -0.018) | No |
| MLP | RS - NR | 0.668 | 0.703 | -0.035 (-0.089 to 0.017) | Yes |
| MLP | FK - NR | 0.643 | 0.703 | -0.060 (-0.129 to 0.006) | Yes |
| MLP | VS - RS | 0.482 | 0.668 | -0.187 (-0.391 to 0.005) | Yes |

## MRI External

| Model | Comparison | AUC arm 1 | AUC arm 2 | Delta AUC (95% CI) | 95% CI contains 0 |
| --- | --- | ---: | ---: | ---: | --- |
| LR | VS - NR | 0.367 | 0.419 | -0.052 (-0.141 to 0.034) | Yes |
| LR | RS - NR | 0.570 | 0.419 | 0.151 (-0.022 to 0.318) | Yes |
| LR | FK - NR | 0.631 | 0.419 | 0.212 (0.006 to 0.416) | No |
| LR | VS - RS | 0.367 | 0.570 | -0.203 (-0.377 to -0.023) | No |
| XGB | VS - NR | 0.417 | 0.456 | -0.039 (-0.199 to 0.119) | Yes |
| XGB | RS - NR | 0.560 | 0.456 | 0.104 (-0.045 to 0.248) | Yes |
| XGB | FK - NR | 0.561 | 0.456 | 0.105 (-0.064 to 0.274) | Yes |
| XGB | VS - RS | 0.417 | 0.560 | -0.142 (-0.324 to 0.039) | Yes |
| MLP | VS - NR | 0.507 | 0.457 | 0.051 (-0.090 to 0.191) | Yes |
| MLP | RS - NR | 0.556 | 0.457 | 0.099 (-0.072 to 0.266) | Yes |
| MLP | FK - NR | 0.675 | 0.457 | 0.218 (0.060 to 0.375) | No |
| MLP | VS - RS | 0.507 | 0.556 | -0.048 (-0.246 to 0.157) | Yes |

## Reviewer 1 Interpretation

For internal performance, formal paired comparisons were performed using paired patient-level bootstrap differences in ROC AUC. Most internal paired delta-AUC confidence intervals included 0, indicating that the observed point-estimate differences should not be interpreted as evidence of superiority. The internal comparisons whose 95% CIs excluded 0 are listed below.

- CT internal, MLP, RS - NR: delta AUC -0.050 (95% CI -0.094 to -0.007).
- CT internal, MLP, VS - RS: delta AUC 0.050 (95% CI 0.007 to 0.093).
- MRI internal, XGB, VS - NR: delta AUC -0.177 (95% CI -0.343 to -0.017).
- MRI internal, XGB, VS - RS: delta AUC -0.249 (95% CI -0.452 to -0.049).
- MRI internal, MLP, VS - NR: delta AUC -0.221 (95% CI -0.429 to -0.018).

Internal comparisons whose 95% CIs included 0:
- CT internal, LR, VS - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- CT internal, LR, RS - NR: delta AUC -0.033 (95% CI -0.072 to 0.004).
- CT internal, LR, FK - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- CT internal, LR, VS - RS: delta AUC 0.033 (95% CI -0.004 to 0.072).
- CT internal, XGB, VS - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- CT internal, XGB, RS - NR: delta AUC -0.052 (95% CI -0.119 to 0.017).
- CT internal, XGB, FK - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- CT internal, XGB, VS - RS: delta AUC 0.052 (95% CI -0.015 to 0.119).
- CT internal, MLP, VS - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- CT internal, MLP, FK - NR: delta AUC 0.000 (95% CI 0.000 to 0.000).
- MRI internal, LR, VS - NR: delta AUC 0.062 (95% CI -0.046 to 0.185).
- MRI internal, LR, RS - NR: delta AUC -0.039 (95% CI -0.233 to 0.162).
- MRI internal, LR, FK - NR: delta AUC -0.065 (95% CI -0.250 to 0.128).
- MRI internal, LR, VS - RS: delta AUC 0.101 (95% CI -0.123 to 0.317).
- MRI internal, XGB, RS - NR: delta AUC 0.071 (95% CI -0.102 to 0.259).
- MRI internal, XGB, FK - NR: delta AUC -0.037 (95% CI -0.212 to 0.130).
- MRI internal, MLP, RS - NR: delta AUC -0.035 (95% CI -0.089 to 0.017).
- MRI internal, MLP, FK - NR: delta AUC -0.060 (95% CI -0.129 to 0.006).
- MRI internal, MLP, VS - RS: delta AUC -0.187 (95% CI -0.391 to 0.005).

## External Cohorts

External paired comparisons were summarized separately. Because the external CT and MRI cohorts are small, these intervals should be interpreted cautiously and not used to claim preprocessing superiority.
External paired delta-AUC 95% CIs excluding 0:
- CT external, LR, RS - NR: delta AUC -0.151 (95% CI -0.278 to -0.025).
- CT external, LR, VS - RS: delta AUC 0.151 (95% CI 0.028 to 0.277).
- CT external, XGB, RS - NR: delta AUC -0.305 (95% CI -0.524 to -0.080).
- CT external, XGB, VS - RS: delta AUC 0.305 (95% CI 0.077 to 0.526).
- MRI external, LR, FK - NR: delta AUC 0.212 (95% CI 0.006 to 0.416).
- MRI external, LR, VS - RS: delta AUC -0.203 (95% CI -0.377 to -0.023).
- MRI external, MLP, FK - NR: delta AUC 0.218 (95% CI 0.060 to 0.375).
