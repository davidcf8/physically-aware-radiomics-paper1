#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leakage-free radiomics modeling across 24 arms.

Design:
- Axis A: NR / RS / VS / FK
- Axis B/N:
    B=OFF,     N=OFF -> no_density
    B=DENSITY, N=OFF -> density
    B=DENSITY, N=ON  -> density_norm
- Axis C:
    OFF / COMBAT

Important principles:
- Source data ALWAYS loaded from:
    <ready_dir>/merged_homogeneous/
- Precomputed combat_transformed CSVs are NOT used for model fitting
- If arm has C=COMBAT:
    * ComBat is learned on TRAIN only
    * During hyperparameter tuning, ComBat is learned fold-wise on inner TRAIN only
- Feature selection is always TRAIN-only
- The previous global ensemble rankings are NOT used for feature selection
  because they may include future test cases and would leak information

Feature modes:
1) COMMON_NEUTRAL
   Build a common comparable signature by consensus across TRAIN-only rankings
   from a reference set of OFF arms (default: all 12 C=OFF arms).
2) COMMON_VS
   Build a common signature by consensus across TRAIN-only VS reference arms only.
3) PER_ARM_ENSEMBLE
   Each arm gets its own TRAIN-only ensemble signature.
4) PER_ARM_MWU
   Each arm gets its own TRAIN-only MWU-FDR signature (baseline simple selector).

Outputs:
- results_models.csv
- preds_test_all.csv
- bundles/
- preds/
- selected_features/
- common_signature/
- run_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    NEUROHARMONIZE_IMPORT_ERROR = None
except Exception as e:
    HAS_NEUROHARMONIZE = False
    NEUROHARMONIZE_IMPORT_ERROR = repr(e)

# XGBoost required
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise SystemExit(
        "xgboost is required for this script.\n"
        f"Import error: {e}"
    )


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
# DATA CLASSES
# =============================================================================
@dataclass
class EnsembleConfig:
    n_select: int = 30
    min_votes: int = 4
    mwu_fdr_alpha: float = 0.05
    spearman_abs_r_min: float = 0.10
    spearman_p_max: float = 0.05
    use_lightgbm: bool = True
    random_state: int = 42


@dataclass
class ModelBundle:
    feature_names: List[str]
    model: object
    model_name: str
    arm_id: str
    threshold: float
    used_combat: bool
    used_feature_mode: str
    selected_features_source: str
    outer_train_size: int
    outer_test_size: int
    combat_batch_mode: Optional[str] = None


@dataclass
class MLPConfig:
    hidden1: int = 64
    hidden2: int = 16
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    val_fraction: float = 0.2
    min_delta_auc: float = 1e-4


@dataclass
class MLPBundle:
    feature_names: List[str]
    model_state_dict: Dict[str, torch.Tensor]
    model_name: str
    arm_id: str
    threshold: float
    used_combat: bool
    used_feature_mode: str
    selected_features_source: str
    outer_train_size: int
    outer_test_size: int
    mlp_params: dict
    imputer_statistics: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    combat_batch_mode: Optional[str] = None

# =============================================================================
# BASIC HELPERS
# =============================================================================

def parse_family_filter_arg(s: Optional[str]) -> Optional[set]:
    if s is None:
        return None
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    if len(vals) == 0:
        return None

    valid = {"firstorder", "shape", "glcm", "glrlm", "gldm", "glszm", "ngtdm", "other"}
    bad = sorted(set(vals) - valid)
    if bad:
        raise ValueError(f"Unknown feature families in filter: {bad}. Valid: {sorted(valid)}")
    return set(vals)


def filter_features_by_family(
    feature_cols: List[str],
    include_families: Optional[set] = None,
    exclude_families: Optional[set] = None,
) -> List[str]:
    out = []
    for f in feature_cols:
        fam = infer_feature_family(f)

        if include_families is not None and fam not in include_families:
            continue
        if exclude_families is not None and fam in exclude_families:
            continue

        out.append(f)
    return out

def sanitize_feature_list_from_train_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
) -> List[str]:
    X_train = train_df[selected_features].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[selected_features].apply(pd.to_numeric, errors="coerce")
    X_train, X_test = sanitize_features_train_test(X_train, X_test)
    return list(X_train.columns)
# =============================================================================
# PYTORCH MLP HELPERS
# =============================================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SmallMLP(nn.Module):
    def __init__(self, n_in: int, hidden1: int = 64, hidden2: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    seed: int,
    device: str = "cpu",
):
    set_all_seeds(seed)

    y_all = y_train.to_numpy(dtype=np.int64)

    if min((y_all == 0).sum(), (y_all == 1).sum()) < 2:
        raise ValueError("MLP requires at least 2 samples per class in training.")

    idx = np.arange(len(y_all))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=params.get("val_fraction", 0.2),
        stratify=y_all,
        random_state=seed,
    )

    # -----------------------------
    # Internal early-stopping split
    # preprocessing fit only on subtrain
    # -----------------------------
    X_subtr_df = X_train.iloc[tr_idx].copy()
    X_val_df = X_train.iloc[va_idx].copy()
    y_subtr = y_all[tr_idx]
    y_val = y_all[va_idx]

    imputer_sub = SimpleImputer(strategy="median")
    scaler_sub = StandardScaler(with_mean=True, with_std=True)

    X_subtr = imputer_sub.fit_transform(X_subtr_df)
    X_subtr = scaler_sub.fit_transform(X_subtr).astype(np.float32)

    X_val = imputer_sub.transform(X_val_df)
    X_val = scaler_sub.transform(X_val).astype(np.float32)

    ds_tr = TabularDataset(X_subtr, y_subtr)
    dl_tr = DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True, drop_last=False)

    model = SmallMLP(
        n_in=X_subtr.shape[1],
        hidden1=params["hidden1"],
        hidden2=params["hidden2"],
        dropout=params["dropout"],
    ).to(device)

    pos = float((y_subtr == 1).sum())
    neg = float((y_subtr == 0).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    best_auc = -np.inf
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        val_prob = predict_mlp_from_array(model, X_val, device=device)
        try:
            val_auc = float(roc_auc_score(y_val, val_prob))
        except Exception:
            val_auc = np.nan

        improved = np.isfinite(val_auc) and (val_auc > best_auc + params.get("min_delta_auc", 1e-4))
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params["patience"]:
            break

    # -----------------------------
    # Final refit on full outer-train
    # -----------------------------
    set_all_seeds(seed)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_full = imputer.fit_transform(X_train)
    X_full = scaler.fit_transform(X_full).astype(np.float32)

    ds_full = TabularDataset(X_full, y_all)
    dl_full = DataLoader(ds_full, batch_size=params["batch_size"], shuffle=True, drop_last=False)

    final_model = SmallMLP(
        n_in=X_full.shape[1],
        hidden1=params["hidden1"],
        hidden2=params["hidden2"],
        dropout=params["dropout"],
    ).to(device)

    pos_full = float((y_all == 1).sum())
    neg_full = float((y_all == 0).sum())
    pos_weight_full = torch.tensor([neg_full / max(1.0, pos_full)], dtype=torch.float32, device=device)

    criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    optim_full = torch.optim.Adam(
        final_model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    epochs_to_train = max(1, int(best_epoch))
    for _ in range(epochs_to_train):
        final_model.train()
        for xb, yb in dl_full:
            xb = xb.to(device)
            yb = yb.to(device)
            optim_full.zero_grad(set_to_none=True)
            logits = final_model(xb)
            loss = criterion_full(logits, yb)
            loss.backward()
            optim_full.step()

    return {
        "imputer": imputer,
        "scaler": scaler,
        "model": final_model,
        "device": device,
        "best_val_auc": float(best_auc) if np.isfinite(best_auc) else np.nan,
        "best_epoch": int(best_epoch),
        "params": params,
    }


def predict_mlp_from_array(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], 256):
            xb = torch.from_numpy(X[start:start + 256]).float().to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)
    return np.concatenate(probs, axis=0)


def predict_mlp_pipeline(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    Xp = fitted["imputer"].transform(X)
    Xp = fitted["scaler"].transform(Xp).astype(np.float32)
    return predict_mlp_from_array(
        model=fitted["model"],
        X=Xp,
        device=fitted.get("device", "cpu"),
    )

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


def ensure_label(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        df["malignancy_label"] = pd.to_numeric(df[target_col], errors="coerce")


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
        raise ValueError("Need malignancy_score or malignancy_label in CSV.")

    return df


def filter_valid_binary_labels(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df["malignancy_label"], errors="coerce")
    keep = y.isin([0, 1])

    dropped = int((~keep).sum())
    if dropped > 0:
        print(f"[WARN] {source_name}: dropping {dropped} rows with invalid malignancy_label")

    df = df.loc[keep].copy()
    if len(df) > 0:
        df["malignancy_label"] = pd.to_numeric(df["malignancy_label"], errors="coerce").astype(int)
    return df


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
        print(f"[WARN] {source_name}: {dup_count} duplicated patient rows found. Keeping first.")
        df = df.drop_duplicates(subset=[key], keep="first").copy()

    return df


def domain_label(B: str, N: str) -> str:
    B = B.upper()
    N = N.upper()
    if B == "OFF" and N == "OFF":
        return "no_density"
    if B == "DENSITY" and N == "OFF":
        return "density"
    if B == "DENSITY" and N == "ON":
        return "density_norm"
    raise ValueError(f"Unsupported B/N combination: {B}/{N}")


def parse_factors_from_path(fp: Path) -> Tuple[str, str, str]:
    name = fp.name.lower()

    if "fake" in name or re.search(r"(^|_)fk(_|\.|$)", name) or "nr_fake" in name:
        A = "FK"
    elif "no_resample" in name or re.search(r"(^|_)nr(_|\.|$)", name):
        A = "NR"
    elif ("resample" in name and "no_resample" not in name) or re.search(r"(^|_)rs(_|\.|$)", name):
        A = "RS"
    elif re.search(r"(^|_)vs(_|\.|$)", name) or "voxelspacing" in name or "voxel_spacing" in name:
        A = "VS"
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


def arm_tag(arm_id: str) -> str:
    return arm_id.replace("|", "__").replace("=", "-")


def infer_feature_family(feature_name: str) -> str:
    f = feature_name.lower()
    for fam, pats in FAMILY_PATTERNS.items():
        if fam == "other":
            continue
        if any(p in f for p in pats):
            return fam
    return "other"


def infer_feature_cols(
    df: pd.DataFrame,
    drop_shape: bool = True,
    include_families: Optional[set] = None,
    exclude_families: Optional[set] = None,
) -> List[str]:
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

    cols = filter_features_by_family(
        cols,
        include_families=include_families,
        exclude_families=exclude_families,
    )
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
# COMBAT HELPERS
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
        raise ImportError(
            f"neuroHarmonize could not be imported. Import error: {NEUROHARMONIZE_IMPORT_ERROR}"
        )

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
# MODEL PREPROCESSING / EVALUATION
# =============================================================================
def fit_lr_pipeline(X_train: pd.DataFrame, y_train: pd.Series, C_value: float, seed: int):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = LogisticRegression(
        C=C_value,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        penalty="l2",
        solver="lbfgs",
    )

    Xtr = imputer.fit_transform(X_train)
    Xtr = scaler.fit_transform(Xtr)
    clf.fit(Xtr, y_train)

    return {"imputer": imputer, "scaler": scaler, "clf": clf}


def predict_lr_pipeline(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    Xp = fitted["imputer"].transform(X)
    Xp = fitted["scaler"].transform(Xp)
    return fitted["clf"].predict_proba(Xp)[:, 1]


def fit_xgb_pipeline(X_train: pd.DataFrame, y_train: pd.Series, params: dict, seed: int):
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / max(1, pos))

    clf = XGBClassifier(
        random_state=seed,
        n_jobs=1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,
        **params,
    )
    clf.fit(Xtr, y_train)
    return {"imputer": imputer, "clf": clf}


def predict_xgb_pipeline(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    Xp = fitted["imputer"].transform(X)
    return fitted["clf"].predict_proba(Xp)[:, 1]


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "auc": np.nan,
        "ap": np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob, n_bins=10)),
        "acc": float((tp + tn) / max(1, len(y_true))),
        "sens": float(tp / max(1, (tp + fn))),
        "spec": float(tn / max(1, (tn + fp))),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    return out


# =============================================================================
# TRAIN-ONLY FEATURE SELECTION
# =============================================================================
def run_train_only_ensemble_ranking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    cfg: EnsembleConfig,
) -> pd.DataFrame:
    X_train = X_train.copy()
    y = y_train.to_numpy(dtype=int)

    # Basic imputation before selectors
    med = X_train.median(axis=0)
    X_train = X_train.fillna(med)

    # Standardize for selectors that benefit from it
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.to_numpy(dtype=float))

    pvals = mwu_pvals(Xs, y)
    p_fdr = bh(pvals)
    sig_mask = (p_fdr < cfg.mwu_fdr_alpha)
    sig_feats = [f for f, keep in zip(feature_names, sig_mask) if keep]

    if len(sig_feats) == 0:
        order = np.argsort(np.nan_to_num(pvals, nan=1.0))
        top = order[:min(len(order), max(cfg.n_select, 50))]
        sig_feats = [feature_names[i] for i in top]

    sig_idx = [feature_names.index(f) for f in sig_feats]
    Xf = Xs[:, sig_idx]

    nf = Xf.shape[1]
    n_select = min(cfg.n_select, nf) if nf > 0 else 0
    if n_select == 0:
        return pd.DataFrame(columns=["feature", "votes", "rank_score", "family"])

    selected: Dict[str, set] = {}

    # RFE
    try:
        rfe = RFE(
            estimator=LogisticRegression(max_iter=2000, solver="liblinear", random_state=cfg.random_state),
            n_features_to_select=n_select
        )
        rfe.fit(Xf, y)
        selected["RFE"] = set(np.array(sig_feats)[rfe.support_].tolist())
    except Exception:
        selected["RFE"] = set()

    # Spearman
    sp_candidates = []
    for j, fname in enumerate(sig_feats):
        try:
            r, p = spearmanr(Xf[:, j], y)
            if np.isfinite(r) and np.isfinite(p) and (abs(r) >= cfg.spearman_abs_r_min) and (p <= cfg.spearman_p_max):
                sp_candidates.append((fname, abs(r)))
        except Exception:
            continue
    sp_candidates.sort(key=lambda t: t[1], reverse=True)
    selected["Spearman"] = set([t[0] for t in sp_candidates[:n_select]])

    # Chi2
    try:
        Xpos = MinMaxScaler().fit_transform(Xf)
        chi_sel = SelectKBest(score_func=chi2, k=n_select)
        chi_sel.fit(Xpos, y)
        selected["Chi2"] = set(np.array(sig_feats)[chi_sel.get_support()].tolist())
    except Exception:
        selected["Chi2"] = set()

    # LogReg coef
    try:
        lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=cfg.random_state)
        lr.fit(Xf, y)
        coefs = np.abs(lr.coef_).flatten()
        top = np.argsort(coefs)[-n_select:]
        selected["LogReg"] = set(np.array(sig_feats)[top].tolist())
    except Exception:
        selected["LogReg"] = set()

    # LightGBM
    if cfg.use_lightgbm and HAS_LGBM:
        try:
            lgbm = LGBMClassifier(n_estimators=300, random_state=cfg.random_state)
            lgbm.fit(Xf, y)
            imp = np.asarray(lgbm.feature_importances_, dtype=float)
            top = np.argsort(imp)[-n_select:]
            selected["LightGBM"] = set(np.array(sig_feats)[top].tolist())
        except Exception:
            selected["LightGBM"] = set()
    else:
        selected["LightGBM"] = set()

    # RF
    try:
        rf = RandomForestClassifier(n_estimators=500, random_state=cfg.random_state, n_jobs=-1)
        rf.fit(Xf, y)
        imp = np.asarray(rf.feature_importances_, dtype=float)
        top = np.argsort(imp)[-n_select:]
        selected["RandomForest"] = set(np.array(sig_feats)[top].tolist())
    except Exception:
        selected["RandomForest"] = set()

    methods = ["RFE", "Spearman", "Chi2", "LogReg", "LightGBM", "RandomForest"]
    rows = []
    for f in feature_names:
        votes = sum(int(f in selected.get(m, set())) for m in methods)
        rows.append({
            "feature": f,
            "votes": votes,
            "rank_score": float(votes),
            "family": infer_feature_family(f),
            **{f"{m}_selected": int(f in selected.get(m, set())) for m in methods},
        })

    out = pd.DataFrame(rows).sort_values(
        ["votes", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    return out


def select_features_train_only_mwu(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    top_k: int,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    X_train = X_train.copy()
    y = y_train.to_numpy(dtype=int)

    med = X_train.median(axis=0)
    X_train = X_train.fillna(med)

    Xv = X_train.to_numpy(dtype=float)
    pvals = mwu_pvals(Xv, y)
    p_fdr = bh(pvals)

    rows = []
    for f, p, pfdr in zip(feature_names, pvals, p_fdr):
        rows.append({
            "feature": f,
            "p_value": p,
            "p_fdr": pfdr,
            "votes": int(np.isfinite(pfdr) and (pfdr < fdr_alpha)),
            "rank_score": float(-np.log10(max(p if np.isfinite(p) else 1.0, 1e-300))),
            "family": infer_feature_family(f),
        })

    out = pd.DataFrame(rows).sort_values(
        ["votes", "p_fdr", "p_value", "feature"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    if int((out["votes"] > 0).sum()) == 0:
        out["votes"] = 1
    return out


def finalize_top_features_from_ranking(
    ranking_df: pd.DataFrame,
    mode: str,
    top_k: int,
    min_votes: int,
) -> List[str]:
    ranking_df = ranking_df.copy()

    if mode == "ENSEMBLE":
        keep = ranking_df[ranking_df["votes"] >= min_votes]["feature"].tolist()
        if len(keep) > 0:
            return keep[:top_k]
        return ranking_df["feature"].head(top_k).tolist()

    if mode == "MWU":
        return ranking_df["feature"].head(top_k).tolist()

    raise ValueError(f"Unsupported ranking mode: {mode}")

def get_all_model_features(
    df: pd.DataFrame,
    drop_shape: bool = True,
    include_families: Optional[set] = None,
    exclude_families: Optional[set] = None,
):
    """
    Return all usable numeric feature columns for modeling.
    Keeps only radiomic features, optionally filtered by family.
    """
    non_feature_cols = {
        "ID", "id",
        "label", "Label", "target", "Target", "y",
        "malignancy", "class", "split",
        "patient_id", "seriesuid", "subset",
        "axisA", "axisB", "axisC",
        "arm", "method", "batch",
        "scanner", "manufacturer", "kernel",
        "fold"
    }

    feature_cols = []
    for col in df.columns:
        if col in non_feature_cols:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        is_radiomic = (
            col.startswith("original_")
            or col.startswith("gradient_")
            or col.startswith("logarithm_")
            or col.startswith("exponential_")
            or col.startswith("wavelet-")
            or col.startswith("lbp-")
        )

        if not is_radiomic:
            continue

        if drop_shape and "_shape_" in col:
            continue

        feature_cols.append(col)

    feature_cols = filter_features_by_family(
        sorted(feature_cols),
        include_families=include_families,
        exclude_families=exclude_families,
    )
    return feature_cols


def sanitize_features_train_test(X_train, X_test):
    """
    Train-only cleaning:
    - remove columns fully missing in train
    - remove constant columns in train
    - fill missing values using train medians
    """
    keep = X_train.columns[X_train.notna().any(axis=0)]
    X_train = X_train[keep].copy()
    X_test = X_test[keep].copy()

    nunique = X_train.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    X_train = X_train[keep].copy()
    X_test = X_test[keep].copy()

    medians = X_train.median(axis=0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test

# =============================================================================
# COMMON SIGNATURE BUILDERS
# =============================================================================
def build_common_signature_from_rankings(
    ranking_map: Dict[str, pd.DataFrame],
    top_k_per_arm: int,
    min_votes: int,
    common_top_k: int,
    min_arm_agreement: int,
    strategy_name: str,
) -> Tuple[List[str], pd.DataFrame]:
    arm_feature_rows = []

    for arm_id, rdf in ranking_map.items():
        selected = finalize_top_features_from_ranking(
            ranking_df=rdf,
            mode="ENSEMBLE",
            top_k=top_k_per_arm,
            min_votes=min_votes,
        )
        sel_df = rdf[rdf["feature"].isin(selected)].copy()

        for _, row in sel_df.iterrows():
            arm_feature_rows.append({
                "arm_id": arm_id,
                "feature": row["feature"],
                "family": row.get("family", infer_feature_family(row["feature"])),
                "votes": float(row.get("votes", 0)),
                "rank_score": float(row.get("rank_score", 0.0)),
            })

    if not arm_feature_rows:
        raise ValueError(f"{strategy_name}: no selected features available to build common signature.")

    tmp = pd.DataFrame(arm_feature_rows)

    agg = (
        tmp.groupby(["feature", "family"], as_index=False)
           .agg(
               arm_count=("arm_id", "nunique"),
               mean_votes=("votes", "mean"),
               mean_rank_score=("rank_score", "mean"),
           )
           .sort_values(
               ["arm_count", "mean_votes", "mean_rank_score", "feature"],
               ascending=[False, False, False, True]
           )
           .reset_index(drop=True)
    )

    eligible = agg[agg["arm_count"] >= min_arm_agreement].copy()
    if eligible.empty:
        eligible = agg.copy()

    common_features = eligible["feature"].head(common_top_k).tolist()
    return common_features, agg


# =============================================================================
# DISCOVERY / LOADING
# =============================================================================
def load_source_csvs(ready_dir: Path,axisA_only: bool = False,target_col: Optional[str] = None,) -> Tuple[Dict[Tuple[str, str, str], pd.DataFrame], pd.DataFrame]:
    merged_dir = ready_dir / "merged_homogeneous"
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


def build_arm_map(source_map: Dict[Tuple[str, str, str], pd.DataFrame], include_combat: bool) -> Dict[str, pd.DataFrame]:
    out = {}
    c_levels = ["OFF"] + (["COMBAT"] if include_combat else [])

    for (A, B, N), df in source_map.items():
        for C in c_levels:
            arm_id = make_arm_id(A, B, N, C)
            out[arm_id] = df.copy()

    return out


def attach_split(df: pd.DataFrame, split_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in arm dataframe.")

    merged = df.merge(split_df[[id_col, "split"]], on=id_col, how="inner")
    if merged.empty:
        raise ValueError("After merging split file, no rows remain.")
    if merged["split"].isna().any():
        raise ValueError("Some IDs have missing split assignment.")
    merged["split"] = merged["split"].astype(str).str.lower()
    return merged


# =============================================================================
# INNER CV TUNING WITH OPTIONAL COMBAT
# =============================================================================
def evaluate_candidate_inner_cv(
    arm_id: str,
    train_df_outer: pd.DataFrame,
    selected_features: List[str],
    model_name: str,
    params: dict,
    batch_mode: str,
    seed: int,
    inner_cv_splits: int,
) -> Tuple[float, float]:
    info = parse_arm_id(arm_id)
    use_combat = (info["C"] == "COMBAT")

    y_outer = train_df_outer["malignancy_label"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed)

    aucs = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y_outer)), y_outer):
        tr_df = train_df_outer.iloc[tr_idx].copy()
        va_df = train_df_outer.iloc[va_idx].copy()

        fold_features = sanitize_feature_list_from_train_df(
            train_df=tr_df,
            test_df=va_df,
            selected_features=selected_features,
        )

        if len(fold_features) == 0:
            continue

        if use_combat:
            tr_df, va_df = apply_combat_train_test(
                train_df=tr_df,
                test_df=va_df,
                feature_cols=fold_features,
                batch_mode=batch_mode,
                min_feature_unique=3,
            )

        Xtr = tr_df[fold_features].apply(pd.to_numeric, errors="coerce")
        Xva = va_df[fold_features].apply(pd.to_numeric, errors="coerce")
        ytr = tr_df["malignancy_label"].astype(int)
        yva = va_df["malignancy_label"].astype(int)

        if model_name == "LR":
            fitted = fit_lr_pipeline(Xtr, ytr, C_value=params["C"], seed=seed)
            prob = predict_lr_pipeline(fitted, Xva)

        elif model_name == "XGB":
            fitted = fit_xgb_pipeline(Xtr, ytr, params=params, seed=seed)
            prob = predict_xgb_pipeline(fitted, Xva)

        elif model_name == "MLP":
            fitted = fit_mlp_pipeline(
                Xtr,
                ytr,
                params=params,
                seed=seed,
                device="cpu",
            )
            prob = predict_mlp_pipeline(fitted, Xva)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        try:
            auc = float(roc_auc_score(yva.to_numpy(), prob))
        except Exception:
            auc = np.nan

        if np.isfinite(auc):
            aucs.append(auc)

    if len(aucs) == 0:
        return np.nan, np.nan

    return float(np.mean(aucs)), float(np.std(aucs))


def tune_model_inner_cv(
    arm_id: str,
    train_df_outer: pd.DataFrame,
    selected_features: List[str],
    model_name: str,
    batch_mode: str,
    seed: int,
    inner_cv_splits: int,
    max_candidates: int,
) -> Tuple[dict, float, float]:
    rng = np.random.RandomState(seed)

    if model_name == "LR":
        grid = [{"C": c} for c in np.logspace(-3, 2, 20)]
        candidates = grid[:max_candidates]

    elif model_name == "XGB":
        full_grid = list(ParameterGrid({
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1, 5, 10],
            "reg_lambda": [0.5, 1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        }))
        if len(full_grid) > max_candidates:
            idx = rng.choice(len(full_grid), size=max_candidates, replace=False)
            candidates = [full_grid[i] for i in idx]
        else:
            candidates = full_grid

    elif model_name == "MLP":
        full_grid = list(ParameterGrid({
            "hidden1": [64, 128],
            "hidden2": [16, 32],
            "dropout": [0.1, 0.2, 0.3],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [32],
            "max_epochs": [200],
            "patience": [20],
            "val_fraction": [0.2],
            "min_delta_auc": [1e-4],
        }))
        if len(full_grid) > max_candidates:
            idx = rng.choice(len(full_grid), size=max_candidates, replace=False)
            candidates = [full_grid[i] for i in idx]
        else:
            candidates = full_grid

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    best_params = None
    best_mean = -np.inf
    best_std = np.nan

    for params in candidates:
        mean_auc, std_auc = evaluate_candidate_inner_cv(
            arm_id=arm_id,
            train_df_outer=train_df_outer,
            selected_features=selected_features,
            model_name=model_name,
            params=params,
            batch_mode=batch_mode,
            seed=seed,
            inner_cv_splits=inner_cv_splits,
        )
        if np.isfinite(mean_auc) and (mean_auc > best_mean):
            best_mean = mean_auc
            best_std = std_auc
            best_params = params

    if best_params is None:
        raise ValueError(f"{arm_id} | {model_name}: failed to tune any valid hyperparameter set.")

    return best_params, float(best_mean), float(best_std)


# =============================================================================
# TRAIN / TEST FOR ONE ARM
# =============================================================================
def train_and_evaluate_one_model(
    arm_id: str,
    train_df_outer: pd.DataFrame,
    test_df_outer: pd.DataFrame,
    selected_features: List[str],
    model_name: str,
    batch_mode: str,
    seed: int,
    inner_cv_splits: int,
    max_candidates: int,
):
    info = parse_arm_id(arm_id)
    use_combat = (info["C"] == "COMBAT")

    best_params, cv_mean, cv_std = tune_model_inner_cv(
        arm_id=arm_id,
        train_df_outer=train_df_outer,
        selected_features=selected_features,
        model_name=model_name,
        batch_mode=batch_mode,
        seed=seed,
        inner_cv_splits=inner_cv_splits,
        max_candidates=max_candidates,
    )

    train_df_fit = train_df_outer.copy()
    test_df_fit = test_df_outer.copy()

    final_features = sanitize_feature_list_from_train_df(
        train_df=train_df_fit,
        test_df=test_df_fit,
        selected_features=selected_features,
    )

    if len(final_features) == 0:
        raise ValueError(f"{arm_id} | {model_name}: no valid features remain after final train/test sanitization.")

    if use_combat:
        train_df_fit, test_df_fit = apply_combat_train_test(
            train_df=train_df_fit,
            test_df=test_df_fit,
            feature_cols=final_features,
            batch_mode=batch_mode,
            min_feature_unique=3,
        )

    Xtr = train_df_fit[final_features].apply(pd.to_numeric, errors="coerce")
    Xte = test_df_fit[final_features].apply(pd.to_numeric, errors="coerce")
    ytr = train_df_fit["malignancy_label"].astype(int)
    yte = test_df_fit["malignancy_label"].astype(int)

    if model_name == "LR":
        fitted = fit_lr_pipeline(Xtr, ytr, C_value=best_params["C"], seed=seed)
        prob_train = predict_lr_pipeline(fitted, Xtr)
        prob_test = predict_lr_pipeline(fitted, Xte)
        fitted_object = fitted

    elif model_name == "XGB":
        fitted = fit_xgb_pipeline(Xtr, ytr, params=best_params, seed=seed)
        prob_train = predict_xgb_pipeline(fitted, Xtr)
        prob_test = predict_xgb_pipeline(fitted, Xte)
        fitted_object = fitted

    elif model_name == "MLP":
        fitted = fit_mlp_pipeline(
            Xtr,
            ytr,
            params=best_params,
            seed=seed,
            device="cpu",
        )
        prob_train = predict_mlp_pipeline(fitted, Xtr)
        prob_test = predict_mlp_pipeline(fitted, Xte)
        fitted_object = fitted

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        train_auc = float(roc_auc_score(ytr.to_numpy(), prob_train))
    except Exception:
        train_auc = np.nan

    test_metrics = evaluate_binary(yte.to_numpy(), prob_test, thr=0.5)

    metrics_row = {
        "arm_id": arm_id,
        "A": info["A"],
        "B": info["B"],
        "N": info["N"],
        "C": info["C"],
        "domain_label": domain_label(info["B"], info["N"]),
        "model": model_name,
        "n_train": int(len(train_df_outer)),
        "n_test": int(len(test_df_outer)),
        "n_features": int(len(final_features)),
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "train_auc": train_auc,
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["ap"],
        "test_brier": test_metrics["brier"],
        "test_ece": test_metrics["ece"],
        "test_acc": test_metrics["acc"],
        "test_sens": test_metrics["sens"],
        "test_spec": test_metrics["spec"],
        "overfit_gap_train_test_auc": (train_auc - test_metrics["auc"]) if np.isfinite(train_auc) and np.isfinite(test_metrics["auc"]) else np.nan,
        "generalization_gap_cv_test_auc": (cv_mean - test_metrics["auc"]) if np.isfinite(cv_mean) and np.isfinite(test_metrics["auc"]) else np.nan,
        "best_params": json.dumps(best_params, sort_keys=True),
    }

    preds_df = pd.DataFrame({
        "patient_id": test_df_outer["patient_id"].astype(str).values,
        "y_true": yte.to_numpy(),
        "y_prob": prob_test,
        "arm_id": arm_id,
        "model": model_name,
    })

    return fitted_object, metrics_row, preds_df, final_features

def resolve_split_source(df: pd.DataFrame, split_df: Optional[pd.DataFrame], id_col: str, arm_id: str) -> pd.DataFrame:
    """
    If split_df is provided, merge external split.
    Otherwise, use the existing 'split' column inside the arm dataframe.
    """
    df = df.copy()

    if split_df is not None:
        if id_col not in df.columns:
            raise ValueError(f"{arm_id}: missing id column '{id_col}'.")

        merged = df.merge(split_df[[id_col, "split"]], on=id_col, how="inner", suffixes=("", "_external"))
        if merged.empty:
            raise ValueError(f"{arm_id}: after merging external split file, no rows remain.")

        if "split_external" in merged.columns:
            merged["split"] = merged["split_external"]
            merged = merged.drop(columns=["split_external"])

        if merged["split"].isna().any():
            raise ValueError(f"{arm_id}: some IDs have missing split assignment after external merge.")

        merged["split"] = merged["split"].astype(str).str.lower().str.strip()
        valid = {"train", "test"}
        bad = sorted(set(merged["split"].unique()) - valid)
        if bad:
            raise ValueError(f"{arm_id}: invalid split labels after external merge: {bad}")

        return merged

    # No external split -> use split already inside CSV
    if "split" not in df.columns:
        raise ValueError(
            f"{arm_id}: no --split_csv was provided and the arm CSV does not contain a 'split' column."
        )

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    valid = {"train", "test"}
    bad = sorted(set(df["split"].dropna().unique()) - valid)
    if bad:
        raise ValueError(f"{arm_id}: invalid split labels found inside CSV: {bad}")

    if df["split"].isna().any():
        raise ValueError(f"{arm_id}: split column contains NaN values.")

    # Optional consistency check by patient_id
    if id_col in df.columns:
        split_per_id = df.groupby(id_col)["split"].nunique(dropna=True)
        inconsistent_ids = split_per_id[split_per_id > 1]
        if len(inconsistent_ids) > 0:
            raise ValueError(
                f"{arm_id}: found patient IDs assigned to more than one split inside the CSV."
            )

    return df
# =============================================================================
# MAIN
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leakage-free radiomics modeling across 24 arms.")

    p.add_argument(
        "--ready_dir",
        type=Path,
        default=Path("<REPO_PARENT>/LIDL_lung/results/ready"),
        help="Directory containing merged_homogeneous/ and optionally combat_transformed/."
    )
    p.add_argument(
    "--split_csv",
    type=Path,
    default=None,
    help="Optional CSV with columns [patient_id, split]. If omitted, the script will use the split column already present inside each arm CSV."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("<REPO_PARENT>/LIDL_lung/study_results/modelado"),
        help="Output directory."
    )
    p.add_argument(
        "--analysis_label",
        type=str,
        default="leakfree_models_24arms",
        help="Subfolder name inside outdir."
    )

    p.add_argument(
        "--include_combat_arms",
        action="store_true",
        help="Include C=COMBAT arms. ComBat is learned train-only and inner-CV-wise."
    )
    p.add_argument(
        "--batch_mode",
        type=str,
        default="from_column",
        choices=["from_column", "manufacturer", "manufacturer_kernel"],
        help="How to define ComBat SITE."
    )
    p.add_argument(
        "--drop_shape",
        action="store_true",
        help="Exclude shape features from modeling."
    )

    p.add_argument(
        "--feature_mode",
        type=str,
        default="COMMON_NEUTRAL",
        choices=["COMMON_NEUTRAL", "COMMON_VS", "PER_ARM_ENSEMBLE", "PER_ARM_MWU", "FULL_FEATURES"],
        help="How to choose the modeling signature."
    )
    p.add_argument(
        "--common_top_k",
        type=int,
        default=30,
        help="Number of features in COMMON signature."
    )
    p.add_argument(
        "--per_arm_top_k",
        type=int,
        default=30,
        help="Max features for per-arm signatures."
    )
    p.add_argument(
        "--min_common_arm_agreement",
        type=int,
        default=4,
        help="Minimum number of reference arms that must include a feature for COMMON modes."
    )
    p.add_argument(
        "--include_mlp",
        action="store_true",
        help="Also train a PyTorch tabular MLP using the exact same leakage-free pipeline."
    )
    p.add_argument(
        "--max_mlp_candidates",
        type=int,
        default=12,
        help="Maximum number of MLP hyperparameter candidates to evaluate in inner CV."
    )
    p.add_argument(
        "--feature_families",
        type=str,
        default=None,
        help=(
            "Comma-separated feature families to include. "
            "Options: firstorder,shape,glcm,glrlm,gldm,glszm,ngtdm,other. "
            "Example: glcm,glrlm,gldm,ngtdm"
        )
    )
    p.add_argument(
        "--exclude_feature_families",
        type=str,
        default=None,
        help=(
            "Comma-separated feature families to exclude. "
            "Example: glszm"
        )
    )

    p.add_argument(
        "--axisA_only",
        action="store_true",
        help="Run only pure geometry comparison: B=OFF, N=OFF, C=OFF."
    )

    p.add_argument(
        "--modality",
        type=str,
        default="CT",
        choices=["CT", "MRI"]
    )

    p.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Binary target column. For MRI use pathological_complete_response."
    )
    

    p.add_argument("--ensemble_n_select", type=int, default=30)
    p.add_argument("--ensemble_min_votes", type=int, default=4)
    p.add_argument("--mwu_fdr_alpha", type=float, default=0.05)
    p.add_argument("--spearman_abs_r_min", type=float, default=0.10)
    p.add_argument("--spearman_p_max", type=float, default=0.05)
    p.add_argument("--no_lightgbm", action="store_true")

    p.add_argument("--inner_cv_splits", type=int, default=5)
    p.add_argument("--max_xgb_candidates", type=int, default=20)
    p.add_argument("--max_lr_candidates", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)

    return p


def main():
    args = build_argparser().parse_args()

    if args.include_combat_arms and not HAS_NEUROHARMONIZE:
        raise SystemExit(
            "You requested --include_combat_arms but neuroHarmonize could not be imported. "
            f"Import error: {NEUROHARMONIZE_IMPORT_ERROR}"
        )

    base_out = args.outdir / args.analysis_label
    (base_out / "bundles").mkdir(parents=True, exist_ok=True)
    (base_out / "preds").mkdir(parents=True, exist_ok=True)
    (base_out / "selected_features").mkdir(parents=True, exist_ok=True)
    (base_out / "common_signature").mkdir(parents=True, exist_ok=True)

    source_map, source_inventory = load_source_csvs(
    args.ready_dir,
    axisA_only=args.axisA_only,
    target_col=args.target_col,)
    source_inventory.to_csv(base_out / "source_inventory.csv", index=False)

    arm_map = build_arm_map(source_map, include_combat=args.include_combat_arms)

    split_df = None
    if args.split_csv is not None:
        split_df = pd.read_csv(args.split_csv)
        if "patient_id" not in split_df.columns and "ID" in split_df.columns:
            split_df = split_df.rename(columns={"ID": "patient_id"})
        if "patient_id" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError("split_csv must contain columns [patient_id, split].")
        split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()

    # Resolve split for every arm
    arm_map = {
        arm_id: resolve_split_source(df, split_df=split_df, id_col="patient_id", arm_id=arm_id)
        for arm_id, df in arm_map.items()
    }

    cfg = EnsembleConfig(
        n_select=args.ensemble_n_select,
        min_votes=args.ensemble_min_votes,
        mwu_fdr_alpha=args.mwu_fdr_alpha,
        spearman_abs_r_min=args.spearman_abs_r_min,
        spearman_p_max=args.spearman_p_max,
        use_lightgbm=(not args.no_lightgbm),
        random_state=args.seed,
    )
    include_families = parse_family_filter_arg(args.feature_families)
    exclude_families = parse_family_filter_arg(args.exclude_feature_families)

    if include_families is not None and exclude_families is not None:
        overlap = sorted(include_families & exclude_families)
        if overlap:
            raise ValueError(f"Families present in both include and exclude filters: {overlap}")
    # -------------------------------------------------------------------------
    # Train-only rankings per arm (for selection only; leakage-free)
    # -------------------------------------------------------------------------
    train_only_rankings_ensemble: Dict[str, pd.DataFrame] = {}
    train_only_rankings_mwu: Dict[str, pd.DataFrame] = {}
    arm_feature_inventory_rows = []
    if args.feature_mode != "FULL_FEATURES":
        for arm_id, df_arm in arm_map.items():
            train_df = df_arm[df_arm["split"] == "train"].copy()
            if train_df.empty:
                raise ValueError(f"{arm_id}: empty training split.")

            feature_cols = infer_feature_cols(
                train_df,
                drop_shape=args.drop_shape,
                include_families=include_families,
                exclude_families=exclude_families,
            )
            if len(feature_cols) == 0:
                raise ValueError(f"{arm_id}: no usable feature columns found.")

            X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
            y_train = train_df["malignancy_label"].astype(int)

            ens_rank = run_train_only_ensemble_ranking(
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_cols,
                cfg=cfg,
            )
            mwu_rank = select_features_train_only_mwu(
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_cols,
                top_k=args.per_arm_top_k,
                fdr_alpha=args.mwu_fdr_alpha,
            )

            train_only_rankings_ensemble[arm_id] = ens_rank
            train_only_rankings_mwu[arm_id] = mwu_rank

            ens_rank.to_csv(base_out / "selected_features" / f"train_only_ensemble_ranking__{arm_tag(arm_id)}.csv", index=False)
            mwu_rank.to_csv(base_out / "selected_features" / f"train_only_mwu_ranking__{arm_tag(arm_id)}.csv", index=False)

            arm_feature_inventory_rows.append({
                "arm_id": arm_id,
                "n_train_rows": len(train_df),
                "n_candidate_features": len(feature_cols),
                "n_ensemble_votes_ge_min": int((ens_rank["votes"] >= args.ensemble_min_votes).sum()),
                "top5_ensemble": ";".join(ens_rank["feature"].head(5).tolist()),
                "top5_mwu": ";".join(mwu_rank["feature"].head(5).tolist()),
            })

        pd.DataFrame(arm_feature_inventory_rows).to_csv(base_out / "arm_feature_inventory.csv", index=False)

    # -------------------------------------------------------------------------
    # Build COMMON signatures if requested
    # -------------------------------------------------------------------------
    common_features = None
    common_summary_df = None

    if args.feature_mode in {"COMMON_NEUTRAL", "COMMON_VS"}:
        if args.feature_mode == "COMMON_NEUTRAL":
            reference_arms = [a for a in arm_map.keys() if parse_arm_id(a)["C"] == "OFF"]
        else:
            reference_arms = [
                a for a in arm_map.keys()
                if parse_arm_id(a)["C"] == "OFF" and parse_arm_id(a)["A"] == "VS"
            ]

        ranking_subset = {a: train_only_rankings_ensemble[a] for a in reference_arms}
        common_features, common_summary_df = build_common_signature_from_rankings(
            ranking_map=ranking_subset,
            top_k_per_arm=args.per_arm_top_k,
            min_votes=args.ensemble_min_votes,
            common_top_k=args.common_top_k,
            min_arm_agreement=args.min_common_arm_agreement,
            strategy_name=args.feature_mode,
        )

        # Must exist in all modeled arms
        modeled_arms = list(arm_map.keys())
        common_features = [
            f for f in common_features
            if all(f in arm_map[a].columns for a in modeled_arms)
        ]

        if len(common_features) == 0:
            raise ValueError(f"{args.feature_mode}: no shared common features remain after column intersection.")

        pd.Series(common_features, name="feature").to_csv(
            base_out / "common_signature" / f"{args.feature_mode.lower()}__features.csv",
            index=False
        )
        common_summary_df.to_csv(
            base_out / "common_signature" / f"{args.feature_mode.lower()}__consensus_table.csv",
            index=False
        )

    # -------------------------------------------------------------------------
    # Model training across all arms
    # -------------------------------------------------------------------------
    all_metrics = []
    all_preds = []

    for arm_id, df_arm in arm_map.items():
        print(f"[INFO] Running arm: {arm_id}")
        train_df = df_arm[df_arm["split"] == "train"].copy()
        test_df = df_arm[df_arm["split"] == "test"].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"{arm_id}: train or test split is empty.")

        if args.feature_mode == "COMMON_NEUTRAL":
            selected_features = common_features
            selected_features_source = "train_only_common_neutral_consensus"
        elif args.feature_mode == "COMMON_VS":
            selected_features = common_features
            selected_features_source = "train_only_common_vs_consensus"
        elif args.feature_mode == "PER_ARM_ENSEMBLE":
            selected_features = finalize_top_features_from_ranking(
                ranking_df=train_only_rankings_ensemble[arm_id],
                mode="ENSEMBLE",
                top_k=args.per_arm_top_k,
                min_votes=args.ensemble_min_votes,
            )
            selected_features_source = "train_only_per_arm_ensemble"
        elif args.feature_mode == "PER_ARM_MWU":
            selected_features = finalize_top_features_from_ranking(
                ranking_df=train_only_rankings_mwu[arm_id],
                mode="MWU",
                top_k=args.per_arm_top_k,
                min_votes=args.ensemble_min_votes,
            )
            selected_features_source = "train_only_per_arm_mwu"
        elif args.feature_mode == "FULL_FEATURES":
            selected_features = get_all_model_features(
                train_df,
                drop_shape=args.drop_shape,
                include_families=include_families,
                exclude_families=exclude_families,
            )
            selected_features_source = "train_only_full_features"
        else:
            raise ValueError(f"Unsupported feature mode: {args.feature_mode}")

        if len(selected_features) == 0:
            raise ValueError(f"{arm_id}: selected feature list is empty.")

        # Train-only sanitization of selected features
        X_train_sel = train_df[selected_features].apply(pd.to_numeric, errors="coerce")
        X_test_sel = test_df[selected_features].apply(pd.to_numeric, errors="coerce")
        X_train_sel, X_test_sel = sanitize_features_train_test(X_train_sel, X_test_sel)
        selected_features = list(X_train_sel.columns)

        if len(selected_features) == 0:
            raise ValueError(f"{arm_id}: no valid selected features remain after sanitization.")

        pd.DataFrame({
            "feature": selected_features,
            "family": [infer_feature_family(f) for f in selected_features],
        }).to_csv(
            base_out / "selected_features" / f"selected_features__{arm_tag(arm_id)}__{args.feature_mode}.csv",
            index=False
        )

        # ---------------------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------------------
        fitted_lr, metrics_lr, preds_lr, final_features_lr = train_and_evaluate_one_model(
            arm_id=arm_id,
            train_df_outer=train_df,
            test_df_outer=test_df,
            selected_features=selected_features,
            model_name="LR",
            batch_mode=args.batch_mode,
            seed=args.seed,
            inner_cv_splits=args.inner_cv_splits,
            max_candidates=args.max_lr_candidates,
        )

        bundle_lr = ModelBundle(
            feature_names=final_features_lr,
            model=fitted_lr,
            model_name="LR",
            arm_id=arm_id,
            threshold=0.5,
            used_combat=(parse_arm_id(arm_id)["C"] == "COMBAT"),
            used_feature_mode=args.feature_mode,
            selected_features_source=selected_features_source,
            outer_train_size=len(train_df),
            outer_test_size=len(test_df),
            combat_batch_mode=args.batch_mode if parse_arm_id(arm_id)["C"] == "COMBAT" else None,
        )
        bundle_lr_path = base_out / "bundles" / f"bundle__LR__{arm_tag(arm_id)}.joblib"
        joblib.dump(bundle_lr, bundle_lr_path)

        preds_lr_path = base_out / "preds" / f"preds_test__LR__{arm_tag(arm_id)}.csv"
        preds_lr.to_csv(preds_lr_path, index=False)

        metrics_lr["bundle_path"] = str(bundle_lr_path)
        metrics_lr["preds_path"] = str(preds_lr_path)
        metrics_lr["selected_features_source"] = selected_features_source
        metrics_lr["feature_mode"] = args.feature_mode
        all_metrics.append(metrics_lr)
        all_preds.append(preds_lr)

        print(
            f"[OK] {arm_id} | LR | n_feat={metrics_lr['n_features']} "
            f"| CV AUC={metrics_lr['cv_auc_mean']:.3f}±{metrics_lr['cv_auc_std']:.3f} "
            f"| Test AUC={metrics_lr['test_auc']:.3f}"
        )

        # ---------------------------------------------------------------------
        # XGBoost
        # ---------------------------------------------------------------------
        fitted_xgb, metrics_xgb, preds_xgb, final_features_xgb = train_and_evaluate_one_model(
            arm_id=arm_id,
            train_df_outer=train_df,
            test_df_outer=test_df,
            selected_features=selected_features,
            model_name="XGB",
            batch_mode=args.batch_mode,
            seed=args.seed,
            inner_cv_splits=args.inner_cv_splits,
            max_candidates=args.max_xgb_candidates,
        )

        bundle_xgb = ModelBundle(
            feature_names=final_features_xgb,
            model=fitted_xgb,
            model_name="XGB",
            arm_id=arm_id,
            threshold=0.5,
            used_combat=(parse_arm_id(arm_id)["C"] == "COMBAT"),
            used_feature_mode=args.feature_mode,
            selected_features_source=selected_features_source,
            outer_train_size=len(train_df),
            outer_test_size=len(test_df),
            combat_batch_mode=args.batch_mode if parse_arm_id(arm_id)["C"] == "COMBAT" else None,
        )
        bundle_xgb_path = base_out / "bundles" / f"bundle__XGB__{arm_tag(arm_id)}.joblib"
        joblib.dump(bundle_xgb, bundle_xgb_path)

        preds_xgb_path = base_out / "preds" / f"preds_test__XGB__{arm_tag(arm_id)}.csv"
        preds_xgb.to_csv(preds_xgb_path, index=False)

        metrics_xgb["bundle_path"] = str(bundle_xgb_path)
        metrics_xgb["preds_path"] = str(preds_xgb_path)
        metrics_xgb["selected_features_source"] = selected_features_source
        metrics_xgb["feature_mode"] = args.feature_mode
        all_metrics.append(metrics_xgb)
        all_preds.append(preds_xgb)

        print(
            f"[OK] {arm_id} | XGB | n_feat={metrics_xgb['n_features']} "
            f"| CV AUC={metrics_xgb['cv_auc_mean']:.3f}±{metrics_xgb['cv_auc_std']:.3f} "
            f"| Test AUC={metrics_xgb['test_auc']:.3f}"
        )

        # ---------------------------------------------------------------------
        # MLP
        # ---------------------------------------------------------------------
        if args.include_mlp:
            fitted_mlp, metrics_mlp, preds_mlp, final_features_mlp = train_and_evaluate_one_model(
                arm_id=arm_id,
                train_df_outer=train_df,
                test_df_outer=test_df,
                selected_features=selected_features,
                model_name="MLP",
                batch_mode=args.batch_mode,
                seed=args.seed,
                inner_cv_splits=args.inner_cv_splits,
                max_candidates=args.max_mlp_candidates,
            )

            bundle_mlp = MLPBundle(
                feature_names=final_features_mlp,
                model_state_dict={k: v.detach().cpu() for k, v in fitted_mlp["model"].state_dict().items()},
                model_name="MLP",
                arm_id=arm_id,
                threshold=0.5,
                used_combat=(parse_arm_id(arm_id)["C"] == "COMBAT"),
                used_feature_mode=args.feature_mode,
                selected_features_source=selected_features_source,
                outer_train_size=len(train_df),
                outer_test_size=len(test_df),
                mlp_params=fitted_mlp["params"],
                imputer_statistics=fitted_mlp["imputer"].statistics_.copy(),
                scaler_mean=fitted_mlp["scaler"].mean_.copy(),
                scaler_scale=fitted_mlp["scaler"].scale_.copy(),
                combat_batch_mode=args.batch_mode if parse_arm_id(arm_id)["C"] == "COMBAT" else None,
            )

            bundle_mlp_path = base_out / "bundles" / f"bundle__MLP__{arm_tag(arm_id)}.joblib"
            joblib.dump(bundle_mlp, bundle_mlp_path)

            preds_mlp_path = base_out / "preds" / f"preds_test__MLP__{arm_tag(arm_id)}.csv"
            preds_mlp.to_csv(preds_mlp_path, index=False)

            metrics_mlp["bundle_path"] = str(bundle_mlp_path)
            metrics_mlp["preds_path"] = str(preds_mlp_path)
            metrics_mlp["selected_features_source"] = selected_features_source
            metrics_mlp["feature_mode"] = args.feature_mode
            all_metrics.append(metrics_mlp)
            all_preds.append(preds_mlp)

            print(
                f"[OK] {arm_id} | MLP | n_feat={metrics_mlp['n_features']} "
                f"| CV AUC={metrics_mlp['cv_auc_mean']:.3f}±{metrics_mlp['cv_auc_std']:.3f} "
                f"| Test AUC={metrics_mlp['test_auc']:.3f}"
            )

    # Save results
    res_df = pd.DataFrame(all_metrics).sort_values(
        by=["test_auc", "cv_auc_mean", "test_ap", "model", "arm_id"],
        ascending=[False, False, False, True, True]
    )
    res_df.to_csv(base_out / "results_models.csv", index=False)

    pred_all = pd.concat(all_preds, axis=0, ignore_index=True)
    pred_all.to_csv(base_out / "preds_test_all.csv", index=False)

    with open(base_out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "ready_dir": str(args.ready_dir),
            "split_csv": str(args.split_csv) if args.split_csv is not None else None,
            "split_source": "external_csv" if args.split_csv is not None else "embedded_in_arm_csvs",
            "outdir": str(base_out),
            "analysis_label": args.analysis_label,
            "include_combat_arms": args.include_combat_arms,
            "batch_mode": args.batch_mode,
            "drop_shape": args.drop_shape,
            "feature_mode": args.feature_mode,
            "common_top_k": args.common_top_k,
            "per_arm_top_k": args.per_arm_top_k,
            "min_common_arm_agreement": args.min_common_arm_agreement,
            "ensemble_n_select": args.ensemble_n_select,
            "ensemble_min_votes": args.ensemble_min_votes,
            "mwu_fdr_alpha": args.mwu_fdr_alpha,
            "spearman_abs_r_min": args.spearman_abs_r_min,
            "spearman_p_max": args.spearman_p_max,
            "use_lightgbm": (not args.no_lightgbm),
            "inner_cv_splits": args.inner_cv_splits,
            "max_xgb_candidates": args.max_xgb_candidates,
            "max_lr_candidates": args.max_lr_candidates,
            "seed": args.seed,
            "source_is_merged_homogeneous_only": True,
            "precomputed_combat_csvs_used_for_training": False,
            "combat_is_train_only_and_inner_cv_wise": True,
            "global_external_rankings_used_for_selection": False,
            "include_mlp": args.include_mlp,
            "max_mlp_candidates": args.max_mlp_candidates,
            "feature_families": args.feature_families,
            "exclude_feature_families": args.exclude_feature_families,
        }, f, indent=2)

    print("\n✅ Done.")
    print(base_out / "results_models.csv")
    print(base_out / "preds_test_all.csv")


if __name__ == "__main__":
    main()


"""
SIN MLP:
-COMMON_NEUTRAL (EL MÁS IMPORTANTE)

python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label common_neutral_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --batch_mode from_column

PER_ARM_ENSEMBLE (techo por brazo):

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label per_arm_ensemble_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode PER_ARM_ENSEMBLE \
  --per_arm_top_k 30 \
  --ensemble_min_votes 4 \
  --batch_mode from_column

FULL_FEATURES (SIN selección):

python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label full_features_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --batch_mode from_column


COMMON_VS (análisis favorable a VS)

python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label common_vs_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode COMMON_VS \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 2 \
  --ensemble_min_votes 4 \
  --batch_mode from_column

PER_ARM_MWU (baseline simple)

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label per_arm_mwu_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode PER_ARM_MWU \
  --per_arm_top_k 30 \
  --batch_mode from_column

  PARTE 2 — Runs con MLP

  COMMON_NEUTRAL + MLP

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label common_neutral_24arms_mlp \
  --include_combat_arms \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --batch_mode from_column \
  --include_mlp

FULL_FEATURES + MLP (MUY IMPORTANTE)

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label full_features_24arms_mlp \
  --include_combat_arms \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --batch_mode from_column \
  --include_mlp

  PER_ARM_ENSEMBLE + MLP (opcional pero útil)

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label per_arm_ensemble_24arms_mlp \
  --include_combat_arms \
  --drop_shape \
  --feature_mode PER_ARM_ENSEMBLE \
  --per_arm_top_k 30 \
  --ensemble_min_votes 4 \
  --batch_mode from_column \
  --include_mlp
  
  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label common_neutral_textures_no_glszm_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --batch_mode from_column \
  --feature_families glcm,glrlm,gldm,ngtdm

  python <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado \
  --analysis_label full_features_textures_no_glszm_24arms \
  --include_combat_arms \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --batch_mode from_column \
  --feature_families glcm,glrlm,gldm,ngtdm
  

4

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA \
  --analysis_label CT_axisA_common_neutral_mlp \
  --modality CT \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --include_mlp \
  --seed 1337

5

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --analysis_label MRI_axisA_common_neutral \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

6

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --analysis_label MRI_axisA_per_arm_ensemble \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode PER_ARM_ENSEMBLE \
  --per_arm_top_k 30 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

  7

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --analysis_label MRI_axisA_full_features \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --inner_cv_splits 5 \
  --seed 1337

  8

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/modeling.py \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --analysis_label MRI_axisA_common_neutral_mlp \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --include_mlp \
  --seed 1337
  """
