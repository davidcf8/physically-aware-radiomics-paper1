# Environment Notes

The original analysis environment was Python 3.10. Release CSV diagnostics in the source extraction tables indicated Python 3.10.12 and common radiomics dependencies, but the modified PyRadiomics implementation itself is intentionally not included here.

Known package versions in the working environment at packaging time included pandas 2.x, NumPy 2.x, scikit-learn, SciPy, statsmodels, xgboost, lightgbm, torch, matplotlib and joblib. Use `requirements.txt` for a compatible runtime.

The Paper 1 release is a downstream analysis package. The voxel-spacing-aware PyRadiomics implementation and extraction code are maintained separately for Paper 2.
