import os
import glob
import numpy as np
import pandas as pd

# =========================
# Config
# =========================
csv_folder = "<REPO_PARENT>/LIDL_lung/results/dummy"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

if not csv_files:
    raise FileNotFoundError("❌ No se encontraron CSVs en la carpeta indicada.")

N_BINS_TARGET = 50                # objetivo típico de bins
PCTL_FALLBACK = 0.75              # fallback: usar percentil 75 del rango
PCTL_OUTLIER = 0.95               # umbral robusto para detectar extremos (bins p95)
BINS_P95_MAX = 100                # si p95 de bins > 100 => demasiado fino
MEDIAN_BINS_MAX = 80              # si mediana de bins > 80 => demasiado fino
MEDIAN_BINS_MIN = 20              # si mediana de bins < 20 => demasiado grueso
MIN_BW = 1e-6                     # evita bw ~ 0 por rangos ~0

# Si quieres forzar siempre p75 (más robusto, más simple), pon esto a True
FORCE_P75 = False


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _compute_bins_stats(values: pd.Series, bw: float) -> dict:
    """
    Estima distribución del número de bins (Range / binWidth) para una columna firstorder_Range.
    """
    if bw is None or not np.isfinite(bw) or bw <= 0:
        return {"mean": np.nan, "std": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}

    bins = values / bw
    bins = bins.replace([np.inf, -np.inf], np.nan).dropna()

    if bins.empty:
        return {"mean": np.nan, "std": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}

    return {
        "mean": _safe_float(bins.mean()),
        "std": _safe_float(bins.std(ddof=1)),
        "p50": _safe_float(bins.quantile(0.50)),
        "p95": _safe_float(bins.quantile(PCTL_OUTLIER)),
        "max": _safe_float(bins.max()),
    }


def _choose_bw(range_values: pd.Series) -> tuple[float, str]:
    """
    Devuelve (binWidth, regla) para una columna '...firstorder_Range' siguiendo:
    - bw_median = median(range)/N_BINS_TARGET
    - si se detectan extremos (p95 bins > BINS_P95_MAX o mediana bins fuera [MEDIAN_BINS_MIN, MEDIAN_BINS_MAX]),
      usar bw_p75 = p75(range)/N_BINS_TARGET
    """
    # Limpieza básica
    rv = range_values.replace([np.inf, -np.inf], np.nan).dropna()
    if rv.empty:
        return (np.nan, "no_data")

    med_range = float(rv.median())
    p75_range = float(rv.quantile(PCTL_FALLBACK))

    # Binwidth candidatos (con mínimo para evitar 0)
    bw_median = max(med_range / N_BINS_TARGET, MIN_BW)
    bw_p75 = max(p75_range / N_BINS_TARGET, MIN_BW)

    if FORCE_P75:
        return (bw_p75, "p75_forced")

    # Evalúa bins con bw_median
    stats_m = _compute_bins_stats(rv, bw_median)

    # Criterio robusto: evita depender de max (outlier único)
    too_many_bins = (np.isfinite(stats_m["p95"]) and stats_m["p95"] > BINS_P95_MAX)
    median_out_of_band = (
        (np.isfinite(stats_m["p50"]) and stats_m["p50"] > MEDIAN_BINS_MAX) or
        (np.isfinite(stats_m["p50"]) and stats_m["p50"] < MEDIAN_BINS_MIN)
    )

    if too_many_bins or median_out_of_band:
        return (bw_p75, "p75_fallback")

    return (bw_median, "median")


def analizar_dataframe(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """
    Imprime una tabla por dataframe y devuelve un resumen con:
    filtro, bw elegido, regla, y stats de bins.
    """
    print(f"\n📊 Análisis de rangos para: {nombre}")

    range_cols = [c for c in df.columns if c.endswith("firstorder_Range")]
    if not range_cols:
        print("⚠️  No se encontraron columnas 'firstorder_Range'.")
        return pd.DataFrame()

    print("\nFiltro                 |    BinWidth | Regla        | Bins(mean) | Bins(std) | Bins(p50) | Bins(p95) | Bins(max)")
    print("-" * 120)

    rows = []
    for col in range_cols:
        # Extrae nombre del filtro (imageType)
        filtro_nombre = col.replace("_firstorder_Range", "")

        bw, rule = _choose_bw(df[col])

        # Stats con el bw elegido (para reportar)
        stats = _compute_bins_stats(df[col].replace([np.inf, -np.inf], np.nan).dropna(), bw)

        # Alertas suaves
        alert = ""
        if np.isfinite(stats["p95"]) and stats["p95"] > BINS_P95_MAX:
            alert += " ⚠️p95>100"
        if np.isfinite(stats["p50"]) and (stats["p50"] > MEDIAN_BINS_MAX or stats["p50"] < MEDIAN_BINS_MIN):
            alert += " ⚠️p50_out"

        print(
            f"{filtro_nombre:<22} | "
            f"{bw:10.6f} | "
            f"{rule:<11} | "
            f"{stats['mean']:9.1f} | "
            f"{stats['std']:8.1f} | "
            f"{stats['p50']:8.1f} | "
            f"{stats['p95']:8.1f} | "
            f"{stats['max']:8.1f}"
            f"{alert}"
        )

        rows.append({
            "filter": filtro_nombre,
            "range_col": col,
            "binWidth": bw,
            "rule": rule,
            "bins_mean": stats["mean"],
            "bins_std": stats["std"],
            "bins_p50": stats["p50"],
            "bins_p95": stats["p95"],
            "bins_max": stats["max"],
        })

    return pd.DataFrame(rows)


# =========================
# Run per CSV + merged
# =========================
dfs = []
summaries = []

for path in csv_files:
    df = pd.read_csv(path)
    nombre = os.path.basename(path)
    dfs.append(df)

    summary = analizar_dataframe(df, nombre)
    if not summary.empty:
        summary["source_csv"] = nombre
        summaries.append(summary)

# Unir todos los CSVs y repetir análisis global
df_total = pd.concat(dfs, ignore_index=True)
summary_total = analizar_dataframe(df_total, "TODOS LOS DATOS UNIDOS")
if not summary_total.empty:
    summary_total["source_csv"] = "ALL_MERGED"
    summaries.append(summary_total)

# Guardar resumen final (opcional)
if summaries:
    df_summary = pd.concat(summaries, ignore_index=True)
    out_path = os.path.join(csv_folder, "binwidth_recommendations_summary.csv")
    df_summary.to_csv(out_path, index=False)
    print(f"\n✅ Resumen guardado en: {out_path}")
else:
    print("\n⚠️ No se generó ningún resumen (no se encontraron columnas firstorder_Range).")
