#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_network_summaries.py
=============================

TAMC–FRANJAMAR Monitor
Generador de resúmenes automáticos de comportamiento colectivo multistación.

Esta versión trabaja SOLO con salidas tabulares reales del pipeline.
No interpreta imágenes.

Usa, cuando existen:

CORE:
    mainshock/sync/sync_multistation.csv
    mainshock/metrics/tamc_24h_metrics_allstations.csv
    mainshock/scan/scan_*_allstations.csv

ADVANCED / CONTEXT:
    mainshock/forzantes/forzantes_*_24h.csv
    mainshock/robust_precursors/null_summary.csv
    mainshock/robust_precursors/precursors_timeseries.csv
    mainshock/robust_precursors/robust_summary.json

Ignora deliberadamente:
    - CSV por estación
    - rot/
    - .npy
    - imágenes

Salida:
    resultados/<EVENTO>/mainshock/network_summary.json
    resultados/network_global_summary.json

Este resumen es descriptivo.
No es predicción, alerta, early warning ni estimación de riesgo.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTADOS_DIR = PROJECT_ROOT / "resultados"
GLOBAL_SUMMARY_NAME = "network_global_summary.json"


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")
    except ValueError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def round_or_none(x: Any, ndigits: int = 6) -> Optional[float]:
    v = safe_float(x)
    if v is None:
        return None
    return round(v, ndigits)


def pct(x: Any, ndigits: int = 2) -> Optional[float]:
    v = safe_float(x)
    if v is None:
        return None
    return round(100.0 * v, ndigits)


def classify_level(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "moderate"
    return "low"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def find_first_csv(folder: Path, patterns: List[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def clean_region_name(event_dir: Path) -> str:
    name = event_dir.name
    name = re.sub(r"_RECENT_\d{8}_\d{6}$", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def safe_read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            out.append(c)
    return out


# ============================================================
# INTERPRETACIÓN POR PLOT / CSV
# ============================================================

def interpret_sync_graph(sync: Dict[str, Any]) -> Dict[str, Any]:
    if not sync.get("available"):
        return {
            "available": False,
            "graph": "multistation synchrony",
            "title": "Multistation synchrony",
            "key_message": "No synchronization CSV available.",
            "reading": "The multistation synchrony plot cannot be interpreted because sync_multistation.csv was not found.",
        }

    n_stations = sync.get("n_stations")
    max_active = sync.get("max_active_count")
    max_frac = sync.get("max_active_frac")
    mean_frac = sync.get("mean_active_frac")
    sync_fraction = sync.get("sync_fraction")
    max_run = sync.get("max_sync_run_samples")
    coherence_level = sync.get("coherence_level")
    temporal_structure = sync.get("temporal_structure")

    if coherence_level == "high" and temporal_structure == "sustained synchronization":
        key = "High and persistent multistation synchrony."
    elif coherence_level == "high":
        key = "High peak synchrony, but sparse or non-persistent."
    elif coherence_level == "moderate":
        key = "Moderate multistation synchrony."
    else:
        key = "Low or sparse multistation synchrony."

    caveat = None
    if coherence_level == "high" and temporal_structure == "sparse synchronization episodes":
        caveat = (
            "The peak value is high, but the low sync fraction and short run length indicate "
            "transient synchronization rather than sustained network-wide coherence."
        )

    return {
        "available": True,
        "graph": "multistation synchrony",
        "title": "Multistation synchrony",
        "source_file": sync.get("source_file"),
        "key_message": key,
        "reading": (
            f"The synchrony CSV shows that up to {max_active} of {n_stations} stations were simultaneously active. "
            f"The maximum active fraction is {max_frac} ({sync.get('max_active_frac_percent')}%), "
            f"whereas the mean active fraction is {mean_frac} ({sync.get('mean_active_frac_percent')}%). "
            f"Synchronized samples represent {sync.get('sync_fraction_percent')}% of the analyzed window, "
            f"with a maximum synchronized run of {max_run} sample(s). "
            f"The temporal structure is classified as '{temporal_structure}'."
        ),
        "technical_interpretation": {
            "coherence_level": coherence_level,
            "temporal_structure": temporal_structure,
            "max_active_fraction_percent": sync.get("max_active_frac_percent"),
            "mean_active_fraction_percent": sync.get("mean_active_frac_percent"),
            "sync_fraction_percent": sync.get("sync_fraction_percent"),
            "max_sync_run_samples": max_run,
        },
        "caveat": caveat,
    }


def interpret_metrics_graph(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not metrics.get("available"):
        return {
            "available": False,
            "graph": "station-resolved z-scores",
            "title": "Station-resolved z-scores",
            "key_message": "No allstations z-score CSV available.",
            "reading": "The z-score plot cannot be interpreted because tamc_24h_metrics_allstations.csv was not found.",
        }

    anomaly_level = metrics.get("anomaly_level")
    max_abs = metrics.get("max_abs_zscore")

    if anomaly_level == "high":
        key = "The global z-score distribution is strongly elevated."
    elif anomaly_level == "moderate":
        key = "The global z-score distribution is moderately elevated."
    else:
        key = "The global z-score distribution remains mostly near background."

    caveat = None
    if anomaly_level == "low" and max_abs is not None and max_abs >= 8:
        caveat = (
            "The background distribution is low by p95 |z|, but at least one isolated extreme outlier is present. "
            "This should not be read as sustained global anomaly."
        )

    return {
        "available": True,
        "graph": "station-resolved z-scores",
        "title": "Station-resolved z-scores",
        "source_file": metrics.get("source_file"),
        "key_message": key,
        "reading": (
            f"The allstations metrics CSV contains {metrics.get('n_rows')} station-time rows. "
            f"The mean |z| is {metrics.get('mean_abs_zscore')}, median |z| is {metrics.get('median_abs_zscore')}, "
            f"p95 |z| is {metrics.get('p95_abs_zscore')}, p99 |z| is {metrics.get('p99_abs_zscore')}, "
            f"and max |z| is {metrics.get('max_abs_zscore')}. "
            f"Using p95 |z|, the global anomaly level is classified as '{anomaly_level}'."
        ),
        "technical_interpretation": {
            "anomaly_level": anomaly_level,
            "mean_abs_zscore": metrics.get("mean_abs_zscore"),
            "median_abs_zscore": metrics.get("median_abs_zscore"),
            "p95_abs_zscore": metrics.get("p95_abs_zscore"),
            "p99_abs_zscore": metrics.get("p99_abs_zscore"),
            "max_abs_zscore": metrics.get("max_abs_zscore"),
            "isolated_outlier_flag": metrics.get("isolated_outlier_flag"),
        },
        "caveat": caveat,
    }


def interpret_scan_graph(scan: Dict[str, Any]) -> Dict[str, Any]:
    if not scan.get("available"):
        return {
            "available": False,
            "graph": "extreme anomaly distribution",
            "title": "Extreme anomaly distribution",
            "key_message": "No allstations scan CSV available.",
            "reading": "The extreme anomaly plot cannot be interpreted because scan_*_allstations.csv was not found.",
        }

    pattern = scan.get("temporal_pattern")
    if pattern == "strongly clustered extremes":
        key = "Extremes are strongly concentrated in time."
    elif pattern == "clustered extremes":
        key = "Extremes show temporal clustering."
    elif pattern == "diffuse or weakly clustered extremes":
        key = "Extremes are present but diffuse or weakly clustered."
    else:
        key = "No clear temporal extreme structure."

    caveat = None
    if pattern in {"clustered extremes", "strongly clustered extremes"}:
        caveat = (
            "Temporal clustering of extremes is meaningful only when read together with synchrony and z-score summaries."
        )

    return {
        "available": True,
        "graph": "extreme anomaly distribution",
        "title": "Extreme anomaly distribution",
        "source_file": scan.get("source_file"),
        "key_message": key,
        "reading": (
            f"The allstations scan CSV identifies {scan.get('n_extremes')} extreme samples. "
            f"The maximum number of extremes in a 30-minute bin is {scan.get('max_extremes_per_30min_bin')}, "
            f"with extremes occupying {scan.get('occupied_30min_bins')} 30-minute bin(s). "
            f"The temporal pattern is classified as '{pattern}'."
        ),
        "technical_interpretation": {
            "n_extremes": scan.get("n_extremes"),
            "kind_counts": scan.get("kind_counts"),
            "max_extremes_per_30min_bin": scan.get("max_extremes_per_30min_bin"),
            "occupied_30min_bins": scan.get("occupied_30min_bins"),
            "temporal_pattern": pattern,
        },
        "caveat": caveat,
    }


def interpret_forcing_graph(forcing: Dict[str, Any]) -> Dict[str, Any]:
    if not forcing.get("available"):
        return {
            "available": False,
            "graph": "anomaly vs synthetic tidal forcing",
            "title": "Anomaly vs synthetic tidal forcing",
            "key_message": "No forcing CSV available.",
            "reading": "The forcing-reference plot cannot be interpreted because forzantes_*_24h.csv was not found.",
        }

    relationship = forcing.get("relationship_to_anomaly")
    if relationship == "weak_or_no_linear_tracking":
        key = "The anomaly does not strongly track the smooth forcing reference."
    elif relationship == "moderate_linear_tracking":
        key = "The anomaly shows moderate linear tracking of the forcing reference."
    elif relationship == "strong_linear_tracking":
        key = "The anomaly strongly tracks the forcing reference."
    else:
        key = "The forcing relationship could not be fully evaluated."

    return {
        "available": True,
        "graph": "anomaly vs synthetic tidal forcing",
        "title": "Anomaly vs synthetic tidal forcing",
        "source_file": forcing.get("source_file"),
        "key_message": key,
        "reading": (
            f"The forcing CSV was read as a contextual reference. "
            f"The strongest absolute correlation between candidate anomaly columns and forcing-like columns is "
            f"{forcing.get('max_abs_correlation')}. "
            f"The relationship is classified as '{relationship}'. "
            f"This curve is treated as a smooth reference, not as a prediction signal."
        ),
        "technical_interpretation": {
            "max_abs_correlation": forcing.get("max_abs_correlation"),
            "best_column_pair": forcing.get("best_column_pair"),
            "relationship_to_anomaly": relationship,
        },
        "caveat": (
            "The forcing curve is contextual only. It should not be interpreted as a forecasting input or causal proof."
        ),
    }


def interpret_robust_graph(robust: Dict[str, Any]) -> Dict[str, Any]:
    if not robust.get("available"):
        return {
            "available": False,
            "graph": "robust precursors / null model",
            "title": "Robustness / null-model comparison",
            "key_message": "No robust precursor files available.",
            "reading": "The robustness layer cannot be interpreted because robust_precursors outputs were not found.",
        }

    significance = robust.get("statistical_support")
    if significance == "above_null":
        key = "The robust layer supports deviation above null expectations."
    elif significance == "consistent_with_null":
        key = "The robust layer is broadly consistent with null expectations."
    else:
        key = "The robust layer was read, but statistical support is not decisive."

    return {
        "available": True,
        "graph": "robust precursors / null model",
        "title": "Robustness / null-model comparison",
        "source_files": robust.get("source_files"),
        "key_message": key,
        "reading": (
            f"The robust precursor layer was evaluated from available CSV/JSON outputs. "
            f"The statistical support is classified as '{significance}'. "
            f"The strongest detected robust value is {robust.get('max_robust_value')}, "
            f"and the strongest null-reference value is {robust.get('max_null_value')}."
        ),
        "technical_interpretation": {
            "statistical_support": significance,
            "max_robust_value": robust.get("max_robust_value"),
            "max_null_value": robust.get("max_null_value"),
            "robust_to_null_ratio": robust.get("robust_to_null_ratio"),
        },
        "caveat": (
            "This is a descriptive robustness check. It does not convert the monitor into a predictive or warning system."
        ),
    }


# ============================================================
# LECTORES CSV CORE
# ============================================================

def read_sync_summary(mainshock_dir: Path) -> Dict[str, Any]:
    sync_dir = mainshock_dir / "sync"

    # Estricto: primero el CSV de sincronía global. No se leen CSV por estación.
    sync_csv = find_first_csv(sync_dir, ["sync_multistation*.csv"])

    out: Dict[str, Any] = {"available": False, "source_file": None}
    if sync_csv is None:
        return out

    try:
        df = pd.read_csv(sync_csv)
    except Exception as e:
        out["error"] = f"Could not read sync CSV: {e}"
        return out

    if df.empty:
        out["error"] = "Empty sync CSV"
        return out

    out["available"] = True
    out["source_file"] = relpath(sync_csv)

    active_count_col = "active_count" if "active_count" in df.columns else None
    active_frac_col = "active_frac" if "active_frac" in df.columns else None
    sync_flag_col = "sync_flag" if "sync_flag" in df.columns else None

    reserved = {"time_center_iso", "time", "active_count", "active_frac", "sync_flag"}
    station_cols = [c for c in df.columns if c not in reserved]

    n_samples = int(len(df))
    n_stations = len(station_cols)

    if active_count_col:
        active_count = pd.to_numeric(df[active_count_col], errors="coerce")
    else:
        station_numeric = df[station_cols].apply(pd.to_numeric, errors="coerce") if station_cols else pd.DataFrame()
        active_count = station_numeric.sum(axis=1) if not station_numeric.empty else pd.Series(dtype=float)

    if active_frac_col:
        active_frac = pd.to_numeric(df[active_frac_col], errors="coerce")
    else:
        active_frac = active_count / max(1, n_stations)

    if sync_flag_col:
        raw = df[sync_flag_col]
        if raw.dtype == bool:
            sync_flag = raw
        else:
            sync_flag = raw.astype(str).str.lower().isin(["true", "1", "yes", "y", "si", "sí"])
    else:
        sync_flag = active_frac >= 0.6

    sync_events = int(sync_flag.sum())
    sync_fraction = float(sync_events / n_samples) if n_samples else 0.0

    active_frac_valid = active_frac.dropna()
    active_count_valid = active_count.dropna()

    mean_active_frac = float(active_frac_valid.mean()) if not active_frac_valid.empty else 0.0
    max_active_frac = float(active_frac_valid.max()) if not active_frac_valid.empty else 0.0
    max_active_count = int(active_count_valid.max()) if not active_count_valid.empty else 0

    max_run = 0
    current = 0
    for val in sync_flag.fillna(False).astype(bool).tolist():
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    if sync_events == 0:
        temporal_structure = "no clear synchronized episodes"
    elif sync_fraction >= 0.10:
        temporal_structure = "sustained synchronization"
    elif max_run >= 5:
        temporal_structure = "clustered synchronization episodes"
    else:
        temporal_structure = "sparse synchronization episodes"

    coherence_level = classify_level(max_active_frac, low=0.40, high=0.70)
    background_network_occupancy = classify_level(mean_active_frac, low=0.10, high=0.30)

    out.update({
        "n_samples": n_samples,
        "n_stations": n_stations,
        "sync_events": sync_events,
        "sync_fraction": round(sync_fraction, 6),
        "sync_fraction_percent": pct(sync_fraction, 4),
        "mean_active_frac": round(mean_active_frac, 6),
        "mean_active_frac_percent": pct(mean_active_frac, 4),
        "max_active_frac": round(max_active_frac, 6),
        "max_active_frac_percent": pct(max_active_frac, 2),
        "max_active_count": max_active_count,
        "max_sync_run_samples": int(max_run),
        "coherence_level": coherence_level,
        "background_network_occupancy": background_network_occupancy,
        "temporal_structure": temporal_structure,
    })

    out["graph_interpretation"] = interpret_sync_graph(out)
    return out


def read_metrics_summary(mainshock_dir: Path) -> Dict[str, Any]:
    # Estricto: solo allstations. Ignora tamc_24h_metrics_BOAB.csv, etc.
    metrics_csv = mainshock_dir / "metrics" / "tamc_24h_metrics_allstations.csv"

    out: Dict[str, Any] = {"available": False, "source_file": None}
    if not metrics_csv.exists():
        return out

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        out["error"] = f"Could not read metrics CSV: {e}"
        return out

    if df.empty:
        out["error"] = "Empty metrics CSV"
        return out

    if "zscore" not in df.columns and "z_score" in df.columns:
        df["zscore"] = df["z_score"]

    if "zscore" not in df.columns:
        out["error"] = "No zscore column found"
        return out

    z = pd.to_numeric(df["zscore"], errors="coerce").dropna()
    if z.empty:
        out["error"] = "No valid zscore values"
        return out

    out["available"] = True
    out["source_file"] = relpath(metrics_csv)

    abs_z = z.abs()
    mean_abs_z = float(abs_z.mean())
    median_abs_z = float(abs_z.median())
    p95_abs_z = float(abs_z.quantile(0.95))
    p99_abs_z = float(abs_z.quantile(0.99))
    max_abs_z = float(abs_z.max())

    if "station_id" in df.columns:
        n_stations = int(df["station_id"].astype(str).nunique())
    else:
        n_stations = None

    anomaly_level = classify_level(p95_abs_z, low=2.0, high=3.0)
    isolated_outlier_flag = bool(anomaly_level == "low" and max_abs_z >= 8.0)

    out.update({
        "n_rows": int(len(df)),
        "n_stations": n_stations,
        "mean_abs_zscore": round(mean_abs_z, 6),
        "median_abs_zscore": round(median_abs_z, 6),
        "p95_abs_zscore": round(p95_abs_z, 6),
        "p99_abs_zscore": round(p99_abs_z, 6),
        "max_abs_zscore": round(max_abs_z, 6),
        "anomaly_level": anomaly_level,
        "isolated_outlier_flag": isolated_outlier_flag,
    })

    out["graph_interpretation"] = interpret_metrics_graph(out)
    return out


def read_scan_summary(mainshock_dir: Path) -> Dict[str, Any]:
    scan_dir = mainshock_dir / "scan"

    # Estricto: solo allstations. Ignora scan_*_BOAB.csv, etc.
    scan_csv = find_first_csv(scan_dir, ["scan_*_allstations.csv", "scan_*_24h_allstations.csv"])

    out: Dict[str, Any] = {"available": False, "source_file": None}
    if scan_csv is None:
        return out

    try:
        df = pd.read_csv(scan_csv)
    except Exception as e:
        out["error"] = f"Could not read scan CSV: {e}"
        return out

    if df.empty:
        out["error"] = "Empty scan CSV"
        return out

    out["available"] = True
    out["source_file"] = relpath(scan_csv)

    n_extremes = int(len(df))
    kind_counts = {}
    if "kind" in df.columns:
        kind_counts = {str(k): int(v) for k, v in df["kind"].value_counts().to_dict().items()}

    temporal_pattern = "not evaluated"
    max_extremes_per_bin = None
    occupied_bins = None
    total_bins = None
    occupied_fraction = None

    if "time_center_iso" in df.columns:
        t = parse_time(df["time_center_iso"])
        tmp = df.copy()
        tmp["time"] = t
        tmp = tmp.dropna(subset=["time"])
        if not tmp.empty:
            tmp["bin"] = tmp["time"].dt.floor("30min")
            counts = tmp.groupby("bin").size()

            max_extremes_per_bin = int(counts.max()) if not counts.empty else 0
            occupied_bins = int((counts > 0).sum()) if not counts.empty else 0

            t_min = tmp["time"].min()
            t_max = tmp["time"].max()
            if pd.notna(t_min) and pd.notna(t_max):
                all_bins = pd.date_range(
                    start=t_min.floor("30min"),
                    end=t_max.ceil("30min"),
                    freq="30min",
                    tz="UTC",
                )
                total_bins = int(len(all_bins)) if len(all_bins) else None
                if total_bins:
                    occupied_fraction = occupied_bins / total_bins

            if max_extremes_per_bin >= 10:
                temporal_pattern = "strongly clustered extremes"
            elif max_extremes_per_bin >= 4:
                temporal_pattern = "clustered extremes"
            elif n_extremes > 0:
                temporal_pattern = "diffuse or weakly clustered extremes"
            else:
                temporal_pattern = "no extremes detected"

    out.update({
        "n_extremes": n_extremes,
        "kind_counts": kind_counts,
        "max_extremes_per_30min_bin": max_extremes_per_bin,
        "occupied_30min_bins": occupied_bins,
        "total_30min_bins_estimated": total_bins,
        "occupied_30min_fraction": round_or_none(occupied_fraction, 6),
        "occupied_30min_fraction_percent": pct(occupied_fraction, 2),
        "temporal_pattern": temporal_pattern,
    })

    out["graph_interpretation"] = interpret_scan_graph(out)
    return out


# ============================================================
# LECTORES ADVANCED / CONTEXT
# ============================================================

def read_forcing_summary(mainshock_dir: Path) -> Dict[str, Any]:
    forcing_dir = mainshock_dir / "forzantes"
    forcing_csv = find_first_csv(forcing_dir, ["forzantes_*_24h.csv", "forzantes*.csv"])

    out: Dict[str, Any] = {"available": False, "source_file": None}
    if forcing_csv is None:
        return out

    try:
        df = pd.read_csv(forcing_csv)
    except Exception as e:
        out["error"] = f"Could not read forcing CSV: {e}"
        return out

    if df.empty:
        out["error"] = "Empty forcing CSV"
        return out

    out["available"] = True
    out["source_file"] = relpath(forcing_csv)

    cols = numeric_columns(df)
    lower_cols = {c: c.lower() for c in cols}

    forcing_candidates = [
        c for c in cols
        if any(k in lower_cols[c] for k in ["tide", "tidal", "marea", "forcing", "forzante", "synthetic"])
    ]
    anomaly_candidates = [
        c for c in cols
        if any(k in lower_cols[c] for k in ["anom", "z", "suscept", "mean", "score"])
        and c not in forcing_candidates
    ]

    # Fallback: si no se detectan nombres, compara columnas numéricas entre sí
    if not forcing_candidates and cols:
        forcing_candidates = cols[-1:]
    if not anomaly_candidates and len(cols) >= 2:
        anomaly_candidates = cols[:-1]

    best_pair = None
    best_corr = None

    for a in anomaly_candidates:
        for f in forcing_candidates:
            if a == f:
                continue
            aa = pd.to_numeric(df[a], errors="coerce")
            ff = pd.to_numeric(df[f], errors="coerce")
            pair = pd.concat([aa, ff], axis=1).dropna()
            if len(pair) < 5:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.isna(corr):
                continue
            corr_abs = abs(float(corr))
            if best_corr is None or corr_abs > best_corr:
                best_corr = corr_abs
                best_pair = {"anomaly_column": a, "forcing_column": f, "correlation": round(float(corr), 6)}

    if best_corr is None:
        relationship = "not_evaluated"
    elif best_corr >= 0.70:
        relationship = "strong_linear_tracking"
    elif best_corr >= 0.40:
        relationship = "moderate_linear_tracking"
    else:
        relationship = "weak_or_no_linear_tracking"

    out.update({
        "numeric_columns": cols,
        "forcing_candidate_columns": forcing_candidates,
        "anomaly_candidate_columns": anomaly_candidates,
        "max_abs_correlation": round_or_none(best_corr, 6),
        "best_column_pair": best_pair,
        "relationship_to_anomaly": relationship,
    })
    out["graph_interpretation"] = interpret_forcing_graph(out)
    return out


def read_robust_summary(mainshock_dir: Path) -> Dict[str, Any]:
    robust_dir = mainshock_dir / "robust_precursors"

    null_csv = robust_dir / "null_summary.csv"
    prec_csv = robust_dir / "precursors_timeseries.csv"
    robust_json = robust_dir / "robust_summary.json"

    out: Dict[str, Any] = {
        "available": False,
        "source_files": {},
    }

    source_files = {}
    if null_csv.exists():
        source_files["null_summary"] = relpath(null_csv)
    if prec_csv.exists():
        source_files["precursors_timeseries"] = relpath(prec_csv)
    if robust_json.exists():
        source_files["robust_summary"] = relpath(robust_json)

    if not source_files:
        return out

    out["available"] = True
    out["source_files"] = source_files

    max_null_value = None
    max_robust_value = None
    robust_json_data = None

    # Leer robust_summary.json si existe
    if robust_json.exists():
        robust_json_data = safe_read_json(robust_json)
        out["robust_summary_json_available"] = robust_json_data is not None

        # Extrae números máximos genéricamente
        nums = []
        def walk(obj: Any):
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
            else:
                val = safe_float(obj)
                if val is not None:
                    nums.append(abs(val))
        if robust_json_data is not None:
            walk(robust_json_data)
            if nums:
                max_robust_value = max(nums)

    # Leer precursors_timeseries.csv
    if prec_csv.exists():
        try:
            dfp = pd.read_csv(prec_csv)
            cols = numeric_columns(dfp)
            vals = []
            for c in cols:
                vals.extend(pd.to_numeric(dfp[c], errors="coerce").dropna().abs().tolist())
            if vals:
                v = max(vals)
                max_robust_value = max(v, max_robust_value) if max_robust_value is not None else v
            out["precursors_rows"] = int(len(dfp))
            out["precursors_numeric_columns"] = cols
        except Exception as e:
            out["precursors_error"] = str(e)

    # Leer null_summary.csv
    if null_csv.exists():
        try:
            dfn = pd.read_csv(null_csv)
            cols = numeric_columns(dfn)
            vals = []
            for c in cols:
                vals.extend(pd.to_numeric(dfn[c], errors="coerce").dropna().abs().tolist())
            if vals:
                max_null_value = max(vals)
            out["null_rows"] = int(len(dfn))
            out["null_numeric_columns"] = cols
        except Exception as e:
            out["null_error"] = str(e)

    ratio = None
    if max_robust_value is not None and max_null_value not in (None, 0):
        ratio = max_robust_value / max_null_value

    # Clasificación conservadora
    if ratio is None:
        statistical_support = "not_evaluated"
    elif ratio >= 1.5:
        statistical_support = "above_null"
    else:
        statistical_support = "consistent_with_null"

    out.update({
        "max_robust_value": round_or_none(max_robust_value, 6),
        "max_null_value": round_or_none(max_null_value, 6),
        "robust_to_null_ratio": round_or_none(ratio, 6),
        "statistical_support": statistical_support,
    })
    out["graph_interpretation"] = interpret_robust_graph(out)
    return out


# ============================================================
# INTERPRETACIÓN CONJUNTA
# ============================================================

def classify_joint_network_state(
    sync: Dict[str, Any],
    metrics: Dict[str, Any],
    scan: Dict[str, Any],
    forcing: Dict[str, Any],
    robust: Dict[str, Any],
) -> str:
    """
    Conservative joint classifier.

    Main principle:
    - A single extreme value must NOT define the network state.
    - The main state is assigned from the joint behaviour of synchrony,
      persistence, global z-score distribution, extreme clustering and
      robust/null support.
    - Extremes are treated as contextual unless they are accompanied by
      coherent and/or persistent multistation structure.
    """
    coherence = sync.get("coherence_level") if sync.get("available") else "not available"
    temporal = sync.get("temporal_structure") if sync.get("available") else "not available"
    anomaly = metrics.get("anomaly_level") if metrics.get("available") else "not available"
    isolated_outlier = bool(metrics.get("isolated_outlier_flag")) if metrics.get("available") else False
    scan_pattern = scan.get("temporal_pattern") if scan.get("available") else "not available"
    robust_support = robust.get("statistical_support") if robust.get("available") else "not_available"

    clustered_extremes = scan_pattern in {"clustered extremes", "strongly clustered extremes"}
    sparse_or_clustered_sync = temporal in {
        "sparse synchronization episodes",
        "clustered synchronization episodes",
    }
    sustained_sync = temporal == "sustained synchronization"
    null_like = robust_support in {"consistent_with_null", "not_evaluated", "not_available", "not available"}
    above_null = robust_support == "above_null"

    # 1) Strong coherent state: requires sustained synchrony and elevated global anomaly.
    if coherence == "high" and sustained_sync and anomaly in {"moderate", "high"}:
        if above_null:
            return "ROBUST COHERENT NETWORK REGIME"
        return "COHERENT NETWORK REGIME"

    # 2) High peak synchrony but not sustained: transient, not a stable regime.
    if coherence == "high" and sparse_or_clustered_sync:
        if anomaly in {"moderate", "high"}:
            if above_null:
                return "ROBUST TRANSIENT COHERENT ANOMALY"
            return "TRANSIENT COHERENT ANOMALY"
        if anomaly == "low" and clustered_extremes:
            if above_null:
                return "ROBUST LOCALIZED TRANSIENT SYNCHRONIZATION"
            return "LOCALIZED TRANSIENT SYNCHRONIZATION"
        return "TRANSIENT MULTISTATION SYNCHRONY"

    # 3) Moderate joint structure with clustered extremes.
    # This is the conservative replacement for over-reading isolated outliers.
    if coherence == "moderate" and anomaly == "low" and clustered_extremes:
        return "MODERATE STRUCTURE WITH CLUSTERED EXTREMES"

    # 4) Moderate network structure without strong global anomaly.
    if coherence == "moderate" and anomaly == "low":
        if isolated_outlier:
            return "MODERATE STRUCTURE WITH ISOLATED EXTREMES"
        if null_like:
            return "MODERATE NULL-LIKE STRUCTURE"
        return "MODERATE NETWORK STRUCTURE"

    if coherence == "moderate" and anomaly == "moderate":
        if clustered_extremes:
            return "MODERATE TRANSITIONAL NETWORK STRUCTURE"
        return "TRANSITIONAL NETWORK REGIME"

    # 5) High anomaly without broad sustained coherence.
    if anomaly == "high" and coherence in {"low", "moderate", "not available"}:
        if clustered_extremes:
            return "LOCALIZED ANOMALY WITH CLUSTERED EXTREMES"
        return "LOCALIZED ANOMALY REGIME"

    # 6) Clustered extremes without broad synchrony: descriptive, not a network-wide regime.
    if clustered_extremes and coherence in {"low", "not available"}:
        return "CLUSTERED EXTREME EPISODES WITHOUT BROAD SYNCHRONY"

    # 7) Isolated outlier: only when there is no coherent joint structure.
    if anomaly == "low" and isolated_outlier and coherence in {"low", "not available"} and not clustered_extremes:
        return "ISOLATED EXTREME OUTLIER EPISODE"

    return "BACKGROUND / LOW COHERENCE"

def compute_joint_score(
    sync: Dict[str, Any],
    metrics: Dict[str, Any],
    scan: Dict[str, Any],
    forcing: Dict[str, Any],
    robust: Dict[str, Any],
) -> Dict[str, Any]:
    score = 0.0
    components = {}

    if sync.get("available"):
        max_frac = safe_float(sync.get("max_active_frac")) or 0.0
        sync_fraction = safe_float(sync.get("sync_fraction")) or 0.0
        max_run = safe_float(sync.get("max_sync_run_samples")) or 0.0

        sync_peak_score = min(30.0, 30.0 * max_frac)
        sync_persistence_score = min(20.0, 200.0 * sync_fraction + min(5.0, max_run))

        score += sync_peak_score + sync_persistence_score
        components["sync_peak_score"] = round(sync_peak_score, 3)
        components["sync_persistence_score"] = round(sync_persistence_score, 3)

    if metrics.get("available"):
        p95 = safe_float(metrics.get("p95_abs_zscore")) or 0.0
        max_abs = safe_float(metrics.get("max_abs_zscore")) or 0.0

        anomaly_distribution_score = min(25.0, (p95 / 3.0) * 25.0)
        isolated_extreme_score = min(10.0, max(0.0, (max_abs - 5.0) / 5.0))

        score += anomaly_distribution_score + isolated_extreme_score
        components["anomaly_distribution_score"] = round(anomaly_distribution_score, 3)
        components["isolated_extreme_score"] = round(isolated_extreme_score, 3)

    if scan.get("available"):
        max_bin = safe_float(scan.get("max_extremes_per_30min_bin")) or 0.0
        scan_cluster_score = min(15.0, max_bin / 10.0)
        score += scan_cluster_score
        components["scan_cluster_score"] = round(scan_cluster_score, 3)

    if robust.get("available") and robust.get("statistical_support") == "above_null":
        robust_score = 10.0
        score += robust_score
        components["robust_support_score"] = robust_score

    # Forzantes NO suma score por sí solo: es contexto, no evidencia de anomalía.
    if forcing.get("available"):
        components["forcing_context_score"] = 0.0

    score = max(0.0, min(100.0, score))

    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "moderate"
    else:
        level = "low"

    return {
        "descriptive_score_0_100": round(score, 2),
        "descriptive_score_level": level,
        "score_components": components,
        "score_notice": (
            "This is a descriptive structure score for UI ranking only. "
            "It is not seismic risk, probability, forecast or warning."
        ),
    }


def build_joint_interpretation(
    region_name: str,
    sync: Dict[str, Any],
    metrics: Dict[str, Any],
    scan: Dict[str, Any],
    forcing: Dict[str, Any],
    robust: Dict[str, Any],
) -> Dict[str, Any]:
    state = classify_joint_network_state(sync, metrics, scan, forcing, robust)
    score = compute_joint_score(sync, metrics, scan, forcing, robust)

    coherence = sync.get("coherence_level", "not available") if sync.get("available") else "not available"
    temporal = sync.get("temporal_structure", "not available") if sync.get("available") else "not available"
    anomaly = metrics.get("anomaly_level", "not available") if metrics.get("available") else "not available"
    scan_pattern = scan.get("temporal_pattern", "not available") if scan.get("available") else "not available"
    forcing_relation = forcing.get("relationship_to_anomaly", "not available") if forcing.get("available") else "not available"
    robust_support = robust.get("statistical_support", "not available") if robust.get("available") else "not available"

    evidence = []
    cross_plot_consistency = []

    if sync.get("available"):
        evidence.append(
            f"Synchrony reaches {sync.get('max_active_count')} of {sync.get('n_stations')} stations at peak, "
            f"but synchronized samples represent {sync.get('sync_fraction_percent')}% of the window."
        )

    if metrics.get("available"):
        evidence.append(
            f"The z-score distribution has p95 |z| = {metrics.get('p95_abs_zscore')} "
            f"and max |z| = {metrics.get('max_abs_zscore')}."
        )

    if scan.get("available"):
        evidence.append(
            f"The extreme scan contains {scan.get('n_extremes')} extremes, with up to "
            f"{scan.get('max_extremes_per_30min_bin')} extremes in a 30-minute bin."
        )

    if forcing.get("available"):
        evidence.append(
            f"The forcing-reference layer shows {forcing_relation} with the anomaly candidates "
            f"(max |corr| = {forcing.get('max_abs_correlation')})."
        )

    if robust.get("available"):
        evidence.append(
            f"The robust/null layer is classified as {robust_support} "
            f"(robust/null ratio = {robust.get('robust_to_null_ratio')})."
        )

    # Lectura conjunta real: tensiones o consistencias entre plots.
    if coherence == "high" and temporal == "sparse synchronization episodes":
        cross_plot_consistency.append(
            "Peak synchrony is high, but persistence is low; this supports transient rather than sustained coherence."
        )

    if anomaly == "low" and metrics.get("isolated_outlier_flag"):
        cross_plot_consistency.append(
            "The global anomaly background is low, but isolated extreme z-score outliers are present."
        )

    if scan_pattern == "strongly clustered extremes" and anomaly == "low":
        cross_plot_consistency.append(
            "Extremes are temporally clustered even though the global z-score distribution remains mostly low."
        )

    if forcing_relation == "weak_or_no_linear_tracking":
        cross_plot_consistency.append(
            "The observed anomaly structure does not simply track the smooth forcing reference."
        )

    if robust_support == "above_null":
        cross_plot_consistency.append(
            "The robust/null layer supports that at least part of the detected structure departs from null expectations."
        )

    if not cross_plot_consistency:
        cross_plot_consistency.append(
            "The available CSV summaries are mutually consistent; no major contradiction between plots is detected."
        )

    if state in {"LOCALIZED TRANSIENT SYNCHRONIZATION", "ROBUST LOCALIZED TRANSIENT SYNCHRONIZATION"}:
        summary = (
            f"{region_name} shows localized transient synchronization: the network can reach high simultaneous "
            "activation, but only in sparse or short-lived episodes. The z-score background is not broadly elevated, "
            "while the extreme scan may show temporal concentration. This indicates short-lived collective structure, "
            "not a sustained network-wide anomaly."
        )
    elif state in {"COHERENT NETWORK REGIME", "ROBUST COHERENT NETWORK REGIME"}:
        summary = (
            f"{region_name} shows a coherent network regime: synchrony is high and persistent, "
            "and station-level anomaly statistics are elevated."
        )
    elif state == "TRANSIENT COHERENT ANOMALY":
        summary = (
            f"{region_name} shows transient coherent anomaly: multistation synchrony appears in episodes "
            "and station-level anomaly statistics are elevated."
        )
    elif state == "MODERATE STRUCTURE WITH CLUSTERED EXTREMES":
        summary = (
            f"{region_name} shows moderate network structure with clustered extremes. "
            "Synchrony and extreme timing suggest episodic structure, but the global z-score distribution remains low. "
            "This should be read as a moderate descriptive pattern, not as a sustained network-wide anomaly."
        )
    elif state == "MODERATE STRUCTURE WITH ISOLATED EXTREMES":
        summary = (
            f"{region_name} shows moderate network structure with isolated extreme values. "
            "The isolated extremes are reported as local or episodic features and do not define the main state by themselves."
        )
    elif state == "MODERATE NULL-LIKE STRUCTURE":
        summary = (
            f"{region_name} shows moderate network structure, but the available robust/null layer does not support "
            "a clear departure from null expectations. This is a conservative descriptive classification."
        )
    elif state == "MODERATE TRANSITIONAL NETWORK STRUCTURE":
        summary = (
            f"{region_name} shows moderate transitional network structure with clustered extremes. "
            "The joint reading indicates an intermediate state rather than a strong coherent regime."
        )
    elif state == "LOCALIZED ANOMALY WITH CLUSTERED EXTREMES":
        summary = (
            f"{region_name} shows localized anomaly behaviour with clustered extremes, "
            "but without evidence for broad sustained multistation coherence."
        )
    elif state == "LOCALIZED ANOMALY REGIME":
        summary = (
            f"{region_name} shows localized anomaly behaviour: z-score anomalies are elevated without broad sustained synchrony."
        )
    elif state == "ISOLATED EXTREME OUTLIER EPISODE":
        summary = (
            f"{region_name} shows an isolated extreme outlier episode on an otherwise low global anomaly background. "
            "This is not interpreted as a sustained network regime."
        )
    else:
        summary = (
            f"{region_name} is classified as {state.lower()} based on the joint reading of synchrony, "
            "z-scores, extreme structure, forcing context and robust/null outputs where available."
        )

    pattern_tags = []
    if coherence != "not available":
        pattern_tags.append(f"coherence: {coherence}")
    if temporal != "not available":
        pattern_tags.append(temporal)
    if anomaly != "not available":
        pattern_tags.append(f"anomaly: {anomaly}")
    if scan_pattern != "not available":
        pattern_tags.append(scan_pattern)
    if forcing_relation != "not available":
        pattern_tags.append(f"forcing: {forcing_relation}")
    if robust_support != "not available":
        pattern_tags.append(f"robust: {robust_support}")
    pattern_tags.append(state)

    return {
        "title": "Joint network interpretation",
        "analysis_type": "Joint multistation coherence / z-score / extreme / forcing / robust-null reading",
        "network_state": state,
        "coherence_level": coherence,
        "anomaly_level": anomaly,
        "temporal_structure": temporal,
        "extreme_structure": scan_pattern,
        "forcing_relationship": forcing_relation,
        "robust_statistical_support": robust_support,
        "pattern_tags": pattern_tags,
        "evidence": evidence,
        "cross_plot_consistency": cross_plot_consistency,
        "summary": summary,
        "interpretation_logic": {
            "sync_used": bool(sync.get("available")),
            "metrics_allstations_used": bool(metrics.get("available")),
            "scan_allstations_used": bool(scan.get("available")),
            "forcing_used_as_context": bool(forcing.get("available")),
            "robust_null_used": bool(robust.get("available")),
            "ignored_station_level_csv": True,
            "ignored_images": True,
            "fusion_method": "rule-based multivariate interpretation from CSV/JSON pipeline outputs",
        },
        **score,
        "notice": (
            "Descriptive network-level analysis only. "
            "Not predictive. Not an early-warning or risk-assessment system."
        ),
    }


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def build_global_summary(event_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []

    for payload in event_payloads:
        interp = payload.get("joint_interpretation") or payload.get("interpretation") or {}
        rows.append({
            "region_name": payload.get("region_name"),
            "event_folder": payload.get("event_folder"),
            "mode": payload.get("mode"),
            "network_state": interp.get("network_state"),
            "descriptive_score_0_100": interp.get("descriptive_score_0_100"),
            "descriptive_score_level": interp.get("descriptive_score_level"),
            "coherence_level": interp.get("coherence_level"),
            "anomaly_level": interp.get("anomaly_level"),
            "temporal_structure": interp.get("temporal_structure"),
            "extreme_structure": interp.get("extreme_structure"),
            "forcing_relationship": interp.get("forcing_relationship"),
            "robust_statistical_support": interp.get("robust_statistical_support"),
            "summary": interp.get("summary"),
        })

    def count_contains(text: str) -> int:
        return sum(1 for r in rows if text.lower() in str(r.get("network_state", "")).lower())

    top = sorted(
        rows,
        key=lambda r: safe_float(r.get("descriptive_score_0_100")) or -1,
        reverse=True,
    )[:10]

    return {
        "schema_version": "3.0",
        "generated_utc": utc_now(),
        "summary_type": "global_multistation_network_coherence_summary",
        "total_regions": len(rows),
        "state_counts": {
            "coherent_network_regime": count_contains("COHERENT NETWORK REGIME"),
            "localized_transient_synchronization": count_contains("LOCALIZED TRANSIENT SYNCHRONIZATION"),
            "transient_coherent_anomaly": count_contains("TRANSIENT COHERENT ANOMALY"),
            "localized_anomaly_regime": count_contains("LOCALIZED ANOMALY"),
            "moderate_structure_with_clustered_extremes": count_contains("MODERATE STRUCTURE WITH CLUSTERED EXTREMES"),
            "moderate_structure_with_isolated_extremes": count_contains("MODERATE STRUCTURE WITH ISOLATED EXTREMES"),
            "isolated_extreme_outlier_episode": count_contains("ISOLATED EXTREME OUTLIER"),
            "background_or_low_coherence": count_contains("BACKGROUND"),
        },
        "top_regions_by_descriptive_score": top,
        "regions": rows,
        "notice": (
            "Global summary is descriptive only. Scores and states are intended for visualization and comparison "
            "of network-structure outputs, not for risk, warning or prediction."
        ),
    }


# ============================================================
# PROCESAR EVENTO
# ============================================================

def process_event_dir(event_dir: Path, overwrite: bool = True) -> Optional[Path]:
    mainshock_dir = event_dir / "mainshock"
    if not mainshock_dir.exists():
        return None

    out_json = mainshock_dir / "network_summary.json"
    if out_json.exists() and not overwrite:
        return out_json

    region_name = clean_region_name(event_dir)

    sync = read_sync_summary(mainshock_dir)
    metrics = read_metrics_summary(mainshock_dir)
    scan = read_scan_summary(mainshock_dir)
    forcing = read_forcing_summary(mainshock_dir)
    robust = read_robust_summary(mainshock_dir)

    graph_interpretations = {
        "sync_multistation": sync.get("graph_interpretation", interpret_sync_graph(sync)),
        "zscore_multistation": metrics.get("graph_interpretation", interpret_metrics_graph(metrics)),
        "extreme_anomaly_distribution": scan.get("graph_interpretation", interpret_scan_graph(scan)),
        "anomaly_vs_synthetic_tidal_forcing": forcing.get("graph_interpretation", interpret_forcing_graph(forcing)),
        "robust_precursors": robust.get("graph_interpretation", interpret_robust_graph(robust)),
    }

    joint_interpretation = build_joint_interpretation(region_name, sync, metrics, scan, forcing, robust)

    payload = {
        "schema_version": "3.0",
        "generated_utc": utc_now(),
        "region_name": region_name,
        "event_folder": event_dir.name,
        "mode": "monitor" if "RECENT" in event_dir.name.upper() else "retrospective",
        "summary_type": "multistation_network_coherence_analysis",
        "data_policy": {
            "uses_images": False,
            "uses_station_level_csv": False,
            "uses_allstations_csv": True,
            "ignored_folders": ["rot"],
            "core_sources": [
                "sync/sync_multistation.csv",
                "metrics/tamc_24h_metrics_allstations.csv",
                "scan/scan_*_allstations.csv",
            ],
            "context_sources": [
                "forzantes/forzantes_*_24h.csv",
                "robust_precursors/null_summary.csv",
                "robust_precursors/precursors_timeseries.csv",
                "robust_precursors/robust_summary.json",
            ],
        },
        "sync": sync,
        "metrics": metrics,
        "scan": scan,
        "forcing": forcing,
        "robust_precursors": robust,
        "graph_interpretations": graph_interpretations,
        "joint_interpretation": joint_interpretation,

        # Alias para mantener compatibilidad con tu app si ya lee payload["interpretation"].
        "interpretation": joint_interpretation,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_json


def read_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate network-level summaries for TAMC-FRANJAMAR monitor outputs."
    )
    parser.add_argument(
        "--resultados",
        type=str,
        default=str(DEFAULT_RESULTADOS_DIR),
        help="Path to resultados directory. Default: ./resultados next to this script.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing network_summary.json files.",
    )
    args = parser.parse_args()

    resultados_dir = Path(args.resultados).expanduser().resolve()
    if not resultados_dir.exists():
        print(f"[ERROR] resultados directory not found: {resultados_dir}")
        return 2

    print("=" * 78)
    print("TAMC–FRANJAMAR · Multistation network coherence summaries V3")
    print("=" * 78)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] resultados:   {resultados_dir}")
    print()

    event_dirs = sorted([p for p in resultados_dir.iterdir() if p.is_dir()])
    if not event_dirs:
        print("[WARN] No event folders found.")
        return 0

    n_ok = 0
    n_skip = 0
    event_payloads: List[Dict[str, Any]] = []

    for event_dir in event_dirs:
        if event_dir.name.lower() in {"otros", "_summary", "earthquake_clustering"}:
            continue

        out = process_event_dir(event_dir, overwrite=not args.no_overwrite)
        if out is None:
            n_skip += 1
            continue

        payload = read_payload(out)
        if payload:
            event_payloads.append(payload)

        n_ok += 1
        interp = (payload or {}).get("joint_interpretation") or (payload or {}).get("interpretation") or {}
        state = interp.get("network_state")
        score = interp.get("descriptive_score_0_100")

        try:
            rel = out.relative_to(PROJECT_ROOT)
        except Exception:
            rel = out

        print(f"[OK] {event_dir.name} -> {rel} | {state} | score={score}")

    global_summary = build_global_summary(event_payloads)
    global_out = resultados_dir / GLOBAL_SUMMARY_NAME
    with open(global_out, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"[DONE] summaries written: {n_ok}")
    print(f"[DONE] skipped/no mainshock: {n_skip}")
    print(f"[DONE] global summary: {global_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
