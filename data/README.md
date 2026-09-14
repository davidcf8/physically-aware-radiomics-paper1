# Data

`data/derived_features/` contains final Paper 1 Axis A derived radiomic feature CSVs only.

Included configurations: NR, RS, VS, FK with B=OFF, N=OFF and C=OFF for CT and MRI, internal and external cohorts.

The data are arranged in `merged_homogeneous/` folders because the original Paper 1 scripts discover inputs from that layout.

Excluded: raw images, masks, density tables, ComBat-transformed tables, backups, temporary files and Paper 4/factorial outputs.

## Licensing and data provenance

Source code in this repository is distributed under the MIT License.

The derived radiomic feature tables are provided for reproducibility of the reported analyses. Their underlying source imaging datasets remain subject to the terms and conditions of their respective data providers. Redistribution of the original images or segmentations is not granted by this repository.

The release does not contain raw CT/MRI images or segmentation masks.

Pseudonymized derived feature tables should not be described as equivalent to fully anonymous source medical data.

Users remain responsible for complying with the applicable terms and governance requirements of the source datasets.

This repository includes row-level derived feature tables for external validation cohorts under `data/derived_features/CT/external/` and `data/derived_features/MRI/external/`. Confirm governance and public-sharing permissions for those derived tables before public distribution.
