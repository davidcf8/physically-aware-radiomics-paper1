# Data Dictionary

## Identifier Columns

- `patient_id`: pseudonymized patient/case identifier used for splits, pairing and bootstrap alignment.
- `annotation_id`: CT nodule annotation identifier when present. Not used as a predictor.

## Outcome Columns

- `malignancy_label`: binary modeling outcome. CT source labels may use benign/malignant strings; MRI uses 0/1 coding.
- `pathological_complete_response`: MRI pCR label when available.
- `split`: `train`, `test`, or `external`.

## Predictors

Radiomic predictors are columns beginning with:

- `original_`: Original image radiomic features, including first-order, shape and texture families.
- `gradient_`: Gradient-filtered radiomic features.

Feature families are encoded in the PyRadiomics-style names, e.g. `original_glcm_*`, `gradient_glszm_*`, `original_shape_*`.

## Columns Not To Use As Predictors

Never use identifier, outcome, split or metadata columns as predictors. The modeling scripts also exclude shape features when `--drop_shape` is used, matching the final predictive modeling runs.

## Privacy Cleanup

The public CSVs exclude direct image/mask paths, DICOM dates, age, sex, scanner metadata and redundant original-ID columns. No raw images or masks are included.
