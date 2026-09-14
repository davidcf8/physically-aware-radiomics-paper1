#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple leakage-free ensemble feature ranking for radiomics modeling.

Goal:
- Produce one CV-wise ranking per arm
- Keep pipeline leakage-free for later modeling
- Allow ComBat only as fold-wise train-only transform
- Keep outputs interpretable and compact

Design:
Axis A (geometry):
    NR, RS, VS, FK

Axis B/N (intensity domain):
    B=OFF,     N=OFF -> no_density
    B=DENSITY, N=OFF -> density
    B=DENSITY, N=ON  -> density_norm

Axis C (harmonization):
    OFF
    COMBAT

Important:
- Source data ALWAYS loaded from:
    results/ready/merged_homogeneous/
- precomputed combat_transformed CSVs are NOT used here
- if C=COMBAT, ComBat is learned inside each fold on TRAIN only

Outputs:
1) rankings_per_arm/
   - one CSV per arm with final feature ranking
2) master_arm_summary.csv
   - one row per arm summarizing stable features and families
3) arm_inventory.csv
4) run_config.json
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# Optional selectors
try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Optional ComBat
try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply  # type: ignore
    HAS_NEUROHARMONIZE = True
except Exception:
    HAS_NEUROHARMONIZE = False


# =============================================================================
# CONSTANTS
# =============================================================================
BENIGN = {1, 2}
MALIGN = {4, 5}

DEFAULT_DROP_COLS = {
    "ID", "patient_id", "subject_id", "case_id", "annotation_id",
    "nodule_id", "malignancy_score", "malignancy_label",
    "method", "A", "B", "C", "N", "arm",
    "split", "batch", "geometry_mode", "intensity_mode",
}

EXCLUDE_PREFIXES = ("diagnostics_",)

META_TO_EXCLUDE = {
    "slice_thickness", "dicom_slice_thickness", "spacing_between_slices", "z_spacing_real",
    "pixel_spacing_x", "pixel_spacing_y", "dicom_pixel_spacing_x", "dicom_pixel_spacing_y",
    "spacing_x", "spacing_y", "spacing_z",
    "voxel_spacing", "diameter", "diameter_mm", "diameter_3d",
    "rows", "columns",
    "manufacturer", "Manufacturer",
    "reconstruction_kernel", "ConvolutionKernel", "kernel", "scanner",
    "patient_age", "patient_sex",
    "acquisition_date", "study_date",
    "kvp", "reconstruction_diameter", "exposure",
}

META_HINTS = (
    "spacing", "thickness", "manufacturer", "kernel", "scanner",
    "patient_", "study_date", "acquisition_date", "rows", "columns",
    "diameter", "batch", "split", "geometry_mode", "intensity_mode",
    "kvp", "exposure", "reconstruction",
)

SHAPE_HINTS = ("shape", "original_shape", "original_shape2d", "shape2d")

FAMILY_PATTERNS = {
    "firstorder": ("_firstorder_",),
    "shape": ("_shape_", "shape2d"),
    "glcm": ("_glcm_",),
    "glrlm": ("_glrlm_",),
    "gldm": ("_gldm_",),
    "glszm": ("_glszm_",),
    "ngtdm": ("_ngtdm_",),
    "other": tuple(),
}


# =============================================================================
# HELPERS
# =============================================================================
def clean_string_series(s: pd.Series) -> pd.Series:
    return s.fillna("UNKNOWN").astype(str).str.strip()


def normalize_manufacturer(s: pd.Series) -> pd.Series:
    s = clean_string_series(s).str.upper()
    replacements = {
        "GE MEDICAL SYSTEMS": "GE",
        "GE HEALTHCARE": "GE",
        "GE": "GE",
        "SIEMENS": "SIEMENS",
        "SIEMENS HEALTHINEERS": "SIEMENS",
        "TOSHIBA": "TOSHIBA",
        "CANON": "CANON",
        "CANON MEDICAL SYSTEMS": "CANON",
        "PHILIPS": "PHILIPS",
        "PHILIPS MEDICAL SYSTEMS": "PHILIPS",
        "UNKNOWN": "UNKNOWN",
    }
    return s.map(lambda x: replacements.get(x, x))


def standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ID": "patient_id",
        "PatientID": "patient_id",

        "Manufacturer": "Manufacturer",
        "manufacturer": "Manufacturer",

        "ConvolutionKernel": "ConvolutionKernel",
        "convolution_kernel": "ConvolutionKernel",
        "Kernel": "ConvolutionKernel",

        "SliceThickness": "dicom_slice_thickness",
        "slice_thickness": "slice_thickness",

        "SpacingBetweenSlices": "spacing_between_slices",
        "Z_Spacing_Real": "z_spacing_real",

        "PixelSpacing_X": "dicom_pixel_spacing_x",
        "PixelSpacing_Y": "dicom_pixel_spacing_y",
        "pixel_spacing_x": "pixel_spacing_x",
        "pixel_spacing_y": "pixel_spacing_y",

        "Rows": "rows",
        "Columns": "columns",

        "PatientSex": "patient_sex",
        "PatientAge": "patient_age",
        "AcquisitionDate": "acquisition_date",
        "StudyDate": "study_date",
        "KVP": "kvp",
        "ReconstructionDiameter": "reconstruction_diameter",
        "Exposure": "exposure",
    }
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=present)


def deduplicate_by_patient(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    key = None
    for c in ("patient_id", "ID"):
        if c in df.columns:
            key = c
            break
    if key is None:
        raise ValueError(f"[{source_name}] missing patient_id/ID column")

    dup_count = df[key].duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] {source_name}: {dup_count} duplicated patient rows found. Keeping first occurrence.")
        df = df.drop_duplicates(subset=[key], keep="first").copy()

    return df


def ensure_label(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        y = pd.to_numeric(df[target_col], errors="coerce")
        df["malignancy_label"] = y
        return df

    label_valid = False

    if "malignancy_label" in df.columns:
        y = pd.to_numeric(df["malignancy_label"], errors="coerce")
        if y.isin([0, 1]).any():
            df["malignancy_label"] = y
            label_valid = True

    if (not label_valid) and ("malignancy_score" in df.columns):
        ms = pd.to_numeric(df["malignancy_score"], errors="coerce")

        def map_score(x):
            if pd.isna(x):
                return np.nan
            x = int(x)
            if x in BENIGN:
                return 0
            if x in MALIGN:
                return 1
            return np.nan

        df["malignancy_label"] = ms.map(map_score)

    if "malignancy_label" not in df.columns:
        raise ValueError("Need malignancy_score, malignancy_label, or --target_col.")

    return df


def filter_valid_binary_labels(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    df = df.copy()

    if "malignancy_label" not in df.columns:
        raise ValueError(f"{source_name}: missing malignancy_label after ensure_label().")

    y = pd.to_numeric(df["malignancy_label"], errors="coerce")
    keep = y.isin([0, 1])

    dropped = int((~keep).sum())
    if dropped > 0:
        print(f"[WARN] {source_name}: dropping {dropped} rows with missing/invalid malignancy_label.")

    df = df.loc[keep].copy()

    if len(df) == 0:
        return df

    df["malignancy_label"] = pd.to_numeric(df["malignancy_label"], errors="coerce").astype(int)
    return df


def validate_domain_choice(B: str, N: str) -> None:
    B = B.upper()
    N = N.upper()
    if B == "OFF" and N != "OFF":
        raise ValueError("Invalid domain: if B=OFF, N must be OFF.")
    if B == "DENSITY" and N not in {"OFF", "ON"}:
        raise ValueError("Invalid domain: for B=DENSITY, N must be OFF or ON.")


def domain_label(B: str, N: str) -> str:
    validate_domain_choice(B, N)
    B = B.upper()
    N = N.upper()

    if B == "OFF":
        return "no_density"
    if B == "DENSITY" and N == "OFF":
        return "density"
    if B == "DENSITY" and N == "ON":
        return "density_norm"
    raise ValueError(f"Unsupported B/N combination: {B}/{N}")


def parse_factors_from_path(fp: Path) -> Tuple[str, str, str]:
    name = fp.name.lower()

    # Detect most specific labels first
    if "fake" in name or "nr_fake" in name or re.search(r"(^|_)fk(_|\.|$)", name):
        A = "FK"
    elif (
        re.search(r"(^|_)vs(_|\.|$)", name)
        or "voxelspacing" in name
        or "voxel_spacing" in name
        or "_vs_" in name
    ):
        A = "VS"
    elif "no_resample" in name or re.search(r"(^|_)nr(_|\.|$)", name):
        A = "NR"
    elif ("resample" in name and "no_resample" not in name) or re.search(r"(^|_)rs(_|\.|$)", name):
        A = "RS"
    else:
        raise ValueError(f"Cannot infer Axis A from filename: {fp.name}")

    if "density_norm" in name:
        B = "DENSITY"
        N = "ON"
    elif re.search(r"(^|_)density(_|\.|$)", name):
        B = "DENSITY"
        N = "OFF"
    else:
        B = "OFF"
        N = "OFF"

    return A, B, N

def make_arm_id(A: str, B: str, N: str, C: str) -> str:
    return f"A={A}|B={B}|N={N}|C={C}"


def parse_arm_id(arm_id: str) -> Dict[str, str]:
    out = {}
    for token in arm_id.split("|"):
        k, v = token.split("=")
        out[k] = v
    return out


def arm_id_to_filename(arm_id: str) -> str:
    d = parse_arm_id(arm_id)
    return f"arm_ranking_A-{d['A']}_B-{d['B']}_N-{d['N']}_C-{d['C']}.csv"


def infer_feature_family(feature_name: str) -> str:
    f = feature_name.lower()
    for fam, pats in FAMILY_PATTERNS.items():
        if fam == "other":
            continue
        if any(p in f for p in pats):
            return fam
    return "other"


def infer_feature_cols(df: pd.DataFrame, drop_shape: bool = True) -> List[str]:
    allowed_prefixes = (
        "original_",
        "gradient_",
        "wavelet-",
        "log-sigma-",
        "square_",
        "squareroot_",
        "logarithm_",
        "exponential_",
        "lbp-2D_",
        "lbp-3D_",
    )

    cols = []
    for c in df.columns:
        lc = c.lower()

        if c in DEFAULT_DROP_COLS:
            continue
        if c in META_TO_EXCLUDE:
            continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(h in lc for h in META_HINTS):
            continue
        if not c.startswith(allowed_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if drop_shape and any(h in lc for h in SHAPE_HINTS):
            continue

        cols.append(c)
    return cols


def bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = ~np.isnan(pvals)
    if mask.sum() == 0:
        return out
    out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return out


def mwu_pvals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    pvals = np.full(p, np.nan, dtype=float)
    for j in range(p):
        try:
            pvals[j] = mannwhitneyu(X[y == 0, j], X[y == 1, j], alternative="two-sided")[1]
        except Exception:
            pvals[j] = np.nan
    return pvals


# =============================================================================
# COMBAT
# =============================================================================
def build_batch_series(df: pd.DataFrame, batch_mode: str = "from_column") -> pd.Series:
    if batch_mode == "from_column" and "batch" in df.columns:
        return clean_string_series(df["batch"])

    manufacturer = (
        normalize_manufacturer(df["Manufacturer"])
        if "Manufacturer" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )

    if batch_mode == "manufacturer":
        return manufacturer

    if batch_mode == "manufacturer_kernel":
        if "ConvolutionKernel" in df.columns and df["ConvolutionKernel"].notna().any():
            kernel = clean_string_series(df["ConvolutionKernel"])
            return manufacturer + "__" + kernel
        return manufacturer

    if batch_mode == "from_column":
        if "ConvolutionKernel" in df.columns and df["ConvolutionKernel"].notna().any():
            kernel = clean_string_series(df["ConvolutionKernel"])
            return manufacturer + "__" + kernel
        return manufacturer

    raise ValueError(f"Unknown batch_mode: {batch_mode}")


def build_combat_covars(df: pd.DataFrame, batch_mode: str = "from_column") -> pd.DataFrame:
    covars = pd.DataFrame(index=df.index)
    covars["SITE"] = build_batch_series(df, batch_mode=batch_mode)

    if "patient_sex" in df.columns:
        sex = clean_string_series(df["patient_sex"]).str.upper()
        sex_map = {"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0, "UNKNOWN": np.nan, "": np.nan}
        covars["patient_sex"] = sex.map(sex_map).astype(float)

    if "patient_age" in df.columns:
        age = clean_string_series(df["patient_age"]).str.upper().str.replace("Y", "", regex=False)
        covars["patient_age"] = pd.to_numeric(age, errors="coerce")

    return covars


def align_and_impute_covars(train_covars: pd.DataFrame, test_covars: pd.DataFrame):
    train_covars = train_covars.copy()
    test_covars = test_covars.copy()

    all_cols = [c for c in train_covars.columns if c in test_covars.columns]
    train_covars = train_covars[all_cols]
    test_covars = test_covars[all_cols]

    for col in all_cols:
        if col == "SITE":
            train_covars[col] = clean_string_series(train_covars[col])
            test_covars[col] = clean_string_series(test_covars[col])
            continue

        train_covars[col] = pd.to_numeric(train_covars[col], errors="coerce")
        test_covars[col] = pd.to_numeric(test_covars[col], errors="coerce")

        med = train_covars[col].median()
        train_covars[col] = train_covars[col].fillna(med)
        test_covars[col] = test_covars[col].fillna(med)

    return train_covars, test_covars


def apply_combat_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    batch_mode: str = "from_column",
    min_feature_unique: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_NEUROHARMONIZE:
        raise ImportError("neuroHarmonize is not installed but COMBAT arm was requested.")

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_features = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    test_features = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    train_feature_medians = train_features.median(axis=0)
    train_features = train_features.fillna(train_feature_medians)
    test_features = test_features.fillna(train_feature_medians)

    valid_cols = []
    for col in train_features.columns:
        x = train_features[col]
        if x.isna().all():
            continue
        if x.nunique(dropna=True) < min_feature_unique:
            continue
        if np.isclose(x.std(ddof=0), 0.0):
            continue
        valid_cols.append(col)

    if len(valid_cols) == 0:
        raise ValueError("No valid feature columns remain for ComBat after filtering.")

    train_features = train_features[valid_cols]
    test_features = test_features[valid_cols]

    train_covars = build_combat_covars(train_df, batch_mode=batch_mode)
    test_covars = build_combat_covars(test_df, batch_mode=batch_mode)
    train_covars, test_covars = align_and_impute_covars(train_covars, test_covars)

    unseen_sites = sorted(set(test_covars["SITE"].unique()) - set(train_covars["SITE"].unique()))
    if unseen_sites:
        raise ValueError(f"Test contains unseen batches for ComBat: {unseen_sites}")

    X_train = train_features.to_numpy(dtype=float)
    X_test = test_features.to_numpy(dtype=float)

    model, X_train_adj = harmonizationLearn(X_train, train_covars)
    X_test_adj = harmonizationApply(X_test, test_covars, model)

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out.loc[:, valid_cols] = X_train_adj
    test_out.loc[:, valid_cols] = X_test_adj

    return train_out, test_out


# =============================================================================
# ENSEMBLE SELECTORS
# =============================================================================
@dataclass
class EnsembleConfig:
    n_select: int = 30
    min_votes: int = 5
    stable_frac: float = 0.6
    mwu_fdr_alpha: float = 0.05
    spearman_abs_r_min: float = 0.10
    spearman_p_max: float = 0.05
    use_lightgbm: bool = True
    random_state: int = 42


def run_ensemble_selectors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    cfg: EnsembleConfig
) -> Dict[str, set]:
    selected: Dict[str, set] = {}

    pvals = mwu_pvals(X_train, y_train)
    p_fdr = bh(pvals)
    sig_mask = (p_fdr < cfg.mwu_fdr_alpha)
    sig_feats = [f for f, keep in zip(feature_names, sig_mask) if keep]

    if len(sig_feats) == 0:
        order = np.argsort(np.nan_to_num(pvals, nan=1.0))
        top = order[:min(len(order), max(cfg.n_select, 50))]
        sig_feats = [feature_names[i] for i in top]

    sig_idx = [feature_names.index(f) for f in sig_feats]
    Xf = X_train[:, sig_idx]

    nf = Xf.shape[1]
    n_select = min(cfg.n_select, nf) if nf > 0 else 0
    if n_select == 0:
        return selected

    try:
        rfe = RFE(
            estimator=LogisticRegression(max_iter=2000, solver="liblinear", random_state=cfg.random_state),
            n_features_to_select=n_select
        )
        rfe.fit(Xf, y_train)
        selected["RFE"] = set(np.array(sig_feats)[rfe.support_].tolist())
    except Exception:
        selected["RFE"] = set()

    sp_candidates = []
    for j, fname in enumerate(sig_feats):
        try:
            r, p = spearmanr(Xf[:, j], y_train)
            if np.isfinite(r) and np.isfinite(p) and (abs(r) >= cfg.spearman_abs_r_min) and (p <= cfg.spearman_p_max):
                sp_candidates.append((fname, abs(r)))
        except Exception:
            continue
    sp_candidates.sort(key=lambda t: t[1], reverse=True)
    selected["Spearman"] = set([t[0] for t in sp_candidates[:n_select]])

    try:
        Xpos = MinMaxScaler().fit_transform(Xf)
        chi_sel = SelectKBest(score_func=chi2, k=n_select)
        chi_sel.fit(Xpos, y_train)
        selected["Chi2"] = set(np.array(sig_feats)[chi_sel.get_support()].tolist())
    except Exception:
        selected["Chi2"] = set()

    try:
        lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=cfg.random_state)
        lr.fit(Xf, y_train)
        coefs = np.abs(lr.coef_).flatten()
        top = np.argsort(coefs)[-n_select:]
        selected["LogReg"] = set(np.array(sig_feats)[top].tolist())
    except Exception:
        selected["LogReg"] = set()

    if cfg.use_lightgbm and HAS_LGBM:
        try:
            lgbm = LGBMClassifier(n_estimators=300, random_state=cfg.random_state)
            lgbm.fit(Xf, y_train)
            imp = np.asarray(lgbm.feature_importances_, dtype=float)
            top = np.argsort(imp)[-n_select:]
            selected["LightGBM"] = set(np.array(sig_feats)[top].tolist())
        except Exception:
            selected["LightGBM"] = set()
    else:
        selected["LightGBM"] = set()

    try:
        rf = RandomForestClassifier(n_estimators=500, random_state=cfg.random_state, n_jobs=-1)
        rf.fit(Xf, y_train)
        imp = np.asarray(rf.feature_importances_, dtype=float)
        top = np.argsort(imp)[-n_select:]
        selected["RandomForest"] = set(np.array(sig_feats)[top].tolist())
    except Exception:
        selected["RandomForest"] = set()

    return selected


# =============================================================================
# LOAD SOURCE DATA
# =============================================================================
def load_source_csvs(
    data_root: Path,
    axisA_only: bool = False,
    target_col: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str, str], pd.DataFrame], pd.DataFrame]:
    merged_dir = data_root / "merged_homogeneous"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Missing directory: {merged_dir}")

    out: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    rows = []

    for fp in sorted(merged_dir.glob("*.csv")):
        try:
            A, B, N = parse_factors_from_path(fp)

            if axisA_only and not (B == "OFF" and N == "OFF"):
                continue
            df = pd.read_csv(fp)
            df = standardize_common_columns(df)
            df = deduplicate_by_patient(df, fp.name)
            df = ensure_label(df, target_col=target_col)
            df = filter_valid_binary_labels(df, source_name=fp.name)

            if len(df) == 0:
                print(f"[WARN] Skipping {fp.name}: no valid labeled rows remain.")
                continue

            if "patient_id" not in df.columns:
                raise ValueError(f"{fp.name} missing patient_id after standardization")

            if df["malignancy_label"].nunique() < 2:
                print(f"[WARN] Skipping {fp.name}: only one class present after filtering.")
                continue

            df["A"] = A
            df["B"] = B
            df["N"] = N

            key = (A, B, N)
            out[key] = df

            rows.append({
                "file": str(fp),
                "A": A,
                "B": B,
                "N": N,
                "domain_label": domain_label(B, N),
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "n_batches": df["batch"].nunique(dropna=True) if "batch" in df.columns else np.nan,
            })

        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue

    if not out:
        raise FileNotFoundError(f"No valid source CSVs found under {merged_dir}")

    inv = pd.DataFrame(rows).sort_values(["B", "N", "A"])
    return out, inv


def build_arm_map(
    source_map: Dict[Tuple[str, str, str], pd.DataFrame],
    include_combat: bool,
) -> Dict[str, pd.DataFrame]:
    out = {}
    c_levels = ["OFF"] + (["COMBAT"] if include_combat else [])

    for (A, B, N), df in source_map.items():
        for C in c_levels:
            arm_id = make_arm_id(A, B, N, C)
            out[arm_id] = df.copy()

    return out


# =============================================================================
# CORE: ONE ARM -> ONE RANKING
# =============================================================================
def run_arm_cv_ranking(
    arm_id: str,
    df: pd.DataFrame,
    out_dir: Path,
    cfg: EnsembleConfig,
    n_splits: int,
    batch_mode: str = "from_column",
    drop_shape: bool = True,
) -> Dict[str, object]:
    info = parse_arm_id(arm_id)
    apply_combat = (info["C"] == "COMBAT")

    feature_cols = infer_feature_cols(df, drop_shape=drop_shape)
    if len(feature_cols) == 0:
        raise ValueError(f"{arm_id}: no usable feature columns found.")

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    y = pd.to_numeric(df["malignancy_label"], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(y) & np.isin(y, [0.0, 1.0])
    df = df.loc[valid_mask].copy()
    X_df = X_df.loc[valid_mask].copy()
    y = pd.to_numeric(df["malignancy_label"], errors="coerce").to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(f"{arm_id}: only one class available.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    methods = ["RFE", "Spearman", "Chi2", "LogReg", "LightGBM", "RandomForest"]
    votes_per_feature = {f: [] for f in feature_cols}
    selected_folds_count = {f: 0 for f in feature_cols}
    method_counts = {m: {f: 0 for f in feature_cols} for m in methods}

    fold_rows = []
    combat_failures = 0

    for fold, (tr, te) in enumerate(skf.split(X_df, y), 1):
        train_df = df.iloc[tr].copy()
        test_df = df.iloc[te].copy()

        Xtr_df = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
        Xte_df = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()

        med = Xtr_df.median(axis=0)
        Xtr_df = Xtr_df.fillna(med)
        Xte_df = Xte_df.fillna(med)

        train_df.loc[:, feature_cols] = Xtr_df.values
        test_df.loc[:, feature_cols] = Xte_df.values

        combat_status = "OFF"
        if apply_combat:
            try:
                train_df, test_df = apply_combat_train_test(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                    batch_mode=batch_mode,
                    min_feature_unique=3,
                )
                combat_status = "ok"
            except Exception as e:
                combat_status = f"error: {e}"
                combat_failures += 1
                fold_rows.append({
                    "fold": fold,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "combat_status": combat_status,
                    "n_selected_features": np.nan,
                })
                continue

        Xtr = train_df[feature_cols].to_numpy(dtype=float)
        ytr = train_df["malignancy_label"].to_numpy(dtype=int)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)

        selected = run_ensemble_selectors(Xtr_s, ytr, feature_cols, cfg)

        votes_this_fold = {}
        for feat in feature_cols:
            v = 0
            for m in methods:
                if feat in selected.get(m, set()):
                    v += 1
                    method_counts[m][feat] += 1
            votes_this_fold[feat] = v
            votes_per_feature[feat].append(v)
            if v >= cfg.min_votes:
                selected_folds_count[feat] += 1

        n_sel = sum(v >= cfg.min_votes for v in votes_this_fold.values())
        fold_rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "combat_status": combat_status,
            "n_selected_features": n_sel,
        })

    ranking_rows = []
    stable_threshold = int(np.ceil(cfg.stable_frac * n_splits))

    for feat in feature_cols:
        votes_list = votes_per_feature[feat]
        if len(votes_list) == 0:
            selected_folds = 0
            selection_freq = 0.0
            mean_votes = 0.0
            max_votes = 0
        else:
            selected_folds = selected_folds_count[feat]
            selection_freq = selected_folds / n_splits
            mean_votes = float(np.mean(votes_list))
            max_votes = int(np.max(votes_list))

        stable = int(selected_folds >= stable_threshold)
        rank_score = float(selection_freq * mean_votes)

        ranking_rows.append({
            "feature": feat,
            "family": infer_feature_family(feat),
            "selected_folds": selected_folds,
            "selection_freq": selection_freq,
            "mean_votes_when_counting_all_folds": mean_votes,
            "max_votes": max_votes,
            "stable": stable,
            "rank_score": rank_score,
            **{f"{m}_count": method_counts[m][feat] for m in methods},
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        ["stable", "rank_score", "selected_folds", "max_votes", "feature"],
        ascending=[False, False, False, False, True]
    )

    rankings_dir = out_dir / "rankings_per_arm"
    folds_dir = out_dir / "fold_logs"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    ranking_df.to_csv(rankings_dir / arm_id_to_filename(arm_id), index=False)
    pd.DataFrame(fold_rows).to_csv(folds_dir / arm_id_to_filename(arm_id).replace("arm_ranking_", "fold_log_"), index=False)

    stable_df = ranking_df[ranking_df["stable"] == 1].copy()
    family_counts = stable_df["family"].value_counts().to_dict()

    return {
        "arm_id": arm_id,
        "A": info["A"],
        "B": info["B"],
        "N": info["N"],
        "C": info["C"],
        "domain_label": domain_label(info["B"], info["N"]),
        "n_rows": len(df),
        "n_features_considered": len(feature_cols),
        "n_stable_features": int((ranking_df["stable"] == 1).sum()),
        "n_selected_once_or_more": int((ranking_df["selected_folds"] > 0).sum()),
        "stable_threshold_folds": stable_threshold,
        "combat_failures": combat_failures,
        "top10_features": ";".join(ranking_df["feature"].head(10).tolist()),
        "top10_stable_features": ";".join(stable_df["feature"].head(10).tolist()),
        "stable_firstorder": int(family_counts.get("firstorder", 0)),
        "stable_glcm": int(family_counts.get("glcm", 0)),
        "stable_glrlm": int(family_counts.get("glrlm", 0)),
        "stable_gldm": int(family_counts.get("gldm", 0)),
        "stable_glszm": int(family_counts.get("glszm", 0)),
        "stable_ngtdm": int(family_counts.get("ngtdm", 0)),
        "stable_shape": int(family_counts.get("shape", 0)),
        "stable_other": int(family_counts.get("other", 0)),
        "stable_texture_total": int(
            family_counts.get("glcm", 0)
            + family_counts.get("glrlm", 0)
            + family_counts.get("gldm", 0)
            + family_counts.get("glszm", 0)
            + family_counts.get("ngtdm", 0)
        ),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple leakage-free ensemble ranking per radiomics arm.")

    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("<REPO_PARENT>/LIDL_lung/results/ready"),
        help="Root ready directory containing merged_homogeneous/."
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("<REPO_PARENT>/LIDL_lung/study_results/ensemble"),
        help="Output directory."
    )
    p.add_argument(
        "--analysis_label",
        type=str,
        default="ensemble_ranking_cv",
        help="Subfolder name inside out_dir."
    )

    p.add_argument(
        "--include_combat",
        action="store_true",
        help="Include C=COMBAT arms. Requires neuroHarmonize."
    )

    p.add_argument("--drop_shape", action="store_true", help="Exclude shape features.")
    p.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--stable_frac", type=float, default=0.6, help="Fraction of folds to mark a feature as stable.")

    p.add_argument("--n_select", type=int, default=30, help="Target selected features per method.")
    p.add_argument("--min_votes", type=int, default=5, help="Minimum method votes within a fold.")
    p.add_argument("--mwu_fdr_alpha", type=float, default=0.05, help="FDR alpha for MWU prefilter.")
    p.add_argument("--spearman_abs_r_min", type=float, default=0.10, help="Minimum |Spearman r|.")
    p.add_argument("--spearman_p_max", type=float, default=0.05, help="Maximum Spearman p-value.")
    p.add_argument("--no_lightgbm", action="store_true", help="Disable LightGBM selector.")
    p.add_argument(
        "--batch_mode",
        type=str,
        default="from_column",
        choices=["from_column", "manufacturer", "manufacturer_kernel"],
        help="How to define ComBat batch/SITE."
    )
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")

    p.add_argument("--axisA_only", action="store_true",
               help="Run only pure geometry comparison: B=OFF, N=OFF, C=OFF.")

    p.add_argument("--modality", type=str, default="CT", choices=["CT", "MRI"])

    p.add_argument("--target_col", type=str, default=None,
                help="Binary outcome column. Default: CT uses malignancy_label/malignancy_score; MRI should pass pCR column.")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.include_combat and not HAS_NEUROHARMONIZE:
        raise SystemExit("You requested --include_combat but neuroHarmonize is not installed.")

    if args.axisA_only and args.include_combat:
        raise SystemExit("For pure Axis A analysis, do not use --include_combat.")

    base_out = args.out_dir / args.analysis_label
    base_out.mkdir(parents=True, exist_ok=True)

    source_map, source_inventory = load_source_csvs(
    args.data_dir,
    axisA_only=args.axisA_only,
    target_col=args.target_col,
    )
    source_inventory.to_csv(base_out / "source_inventory.csv", index=False)

    arm_map = build_arm_map(source_map, include_combat=args.include_combat)

    arm_inventory_rows = []
    for arm_id, df in arm_map.items():
        info = parse_arm_id(arm_id)
        arm_inventory_rows.append({
            "arm_id": arm_id,
            **info,
            "domain_label": domain_label(info["B"], info["N"]),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "combat_foldwise_train_only": (info["C"] == "COMBAT"),
        })
    pd.DataFrame(arm_inventory_rows).sort_values(["B", "N", "C", "A"]).to_csv(
        base_out / "arm_inventory.csv",
        index=False
    )

    cfg = EnsembleConfig(
        n_select=args.n_select,
        min_votes=args.min_votes,
        stable_frac=args.stable_frac,
        mwu_fdr_alpha=args.mwu_fdr_alpha,
        spearman_abs_r_min=args.spearman_abs_r_min,
        spearman_p_max=args.spearman_p_max,
        use_lightgbm=(not args.no_lightgbm),
        random_state=args.random_state,
    )

    summary_rows = []
    for arm_id, df in arm_map.items():
        print(f"[INFO] Running {arm_id} ...")
        try:
            row = run_arm_cv_ranking(
                arm_id=arm_id,
                df=df,
                out_dir=base_out,
                cfg=cfg,
                n_splits=args.n_splits,
                batch_mode=args.batch_mode,
                drop_shape=args.drop_shape,
            )
            summary_rows.append(row)
            print(f"[OK] {arm_id}")
        except Exception as e:
            print(f"[WARN] Failed {arm_id}: {e}")
            info = parse_arm_id(arm_id)
            summary_rows.append({
                "arm_id": arm_id,
                "A": info["A"],
                "B": info["B"],
                "N": info["N"],
                "C": info["C"],
                "domain_label": domain_label(info["B"], info["N"]),
                "n_rows": len(df),
                "n_features_considered": np.nan,
                "n_stable_features": np.nan,
                "n_selected_once_or_more": np.nan,
                "stable_threshold_folds": np.nan,
                "combat_failures": np.nan,
                "top10_features": "",
                "top10_stable_features": "",
                "stable_firstorder": np.nan,
                "stable_glcm": np.nan,
                "stable_glrlm": np.nan,
                "stable_gldm": np.nan,
                "stable_glszm": np.nan,
                "stable_ngtdm": np.nan,
                "stable_shape": np.nan,
                "stable_other": np.nan,
                "stable_texture_total": np.nan,
                "error": str(e),
            })

    master = pd.DataFrame(summary_rows).sort_values(["B", "N", "C", "A"])
    master.to_csv(base_out / "master_arm_summary.csv", index=False)

    with open(base_out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir": str(args.data_dir),
            "out_dir": str(base_out),
            "analysis_label": args.analysis_label,
            "include_combat": args.include_combat,
            "drop_shape": args.drop_shape,
            "n_splits": args.n_splits,
            "stable_frac": args.stable_frac,
            "n_select": args.n_select,
            "min_votes": args.min_votes,
            "mwu_fdr_alpha": args.mwu_fdr_alpha,
            "spearman_abs_r_min": args.spearman_abs_r_min,
            "spearman_p_max": args.spearman_p_max,
            "batch_mode": args.batch_mode,
            "use_lightgbm": (not args.no_lightgbm),
            "random_state": args.random_state,
            "source_is_merged_homogeneous_only": True,
            "combat_is_foldwise_train_only": True,
            "precomputed_combat_csvs_used": False,
        }, f, indent=2)

    print("✅ Done.")

    """python <REPO_PARENT>/LIDL_lung/scripts/analisis/ensemble.py \
  --data_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --out_dir <REPO_PARENT>/LIDL_lung/study_results/ensemble \
  --analysis_label ranking_full_24arms \
  --drop_shape \
  --include_combat \
  --batch_mode from_column"""
