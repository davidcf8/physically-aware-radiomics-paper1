# Reproducibility

Run the release check:

```bash
python scripts/validation/reproducibility_check.py
```

Expected checks:

- CT internal cohort: 673 patients; split 538/135.
- MRI internal cohort: 208 patients; split 163/45.
- CT external cohort: 46 patients.
- MRI external cohort: 63 patients.
- Final per-arm predictive feature counts: CT NR/RS/VS/FK = 5/8/12/9; MRI NR/RS/VS/FK = 16/13/12/10.
- Auxiliary relaxed recurrence counts: CT = 7/11/6/7; MRI = 21/26/20/22.
- Paired internal bootstrap: 24 comparisons, 19 CIs include zero, 5 exclude zero.

Full model retraining is possible from the derived feature CSVs but may require substantial runtime and installed optional ML dependencies. Public release does not include model binaries.
