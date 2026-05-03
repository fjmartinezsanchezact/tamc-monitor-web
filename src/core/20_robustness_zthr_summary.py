#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_robustness_zthr_summary.py

Genera figura de robustez (métrica vs zthr) para un evento:
- Real event: leído de C_robustness/zthr_*/real_multi_phase_summary.csv
- Null A / Null B: leído de nullA_simulations.csv / nullB_simulations.csv (si existen)
  con bandas p5–p95 y medianas por (event, zthr).

Uso típico (desde src/core):
  python 20_robustness_zthr_summary.py --event-substr Tohoku --metric n_phases

Si quieres fijar el evento exacto:
  python 20_robustness_zthr_summary.py --event "2011_Great_Tohoku_Earthquake_Japan_M9.1_20110311_054624" --metric n_phases

Notas:
- Si tu CSV combinado tiene solo un zthr, el plot tendrá solo un punto. Este script
  evita eso leyendo directamente zthr_3..6 (si están disponibles).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _repo_root_from_this_file() -> Path:
    # .../tamcsismico/src/core/20_robustness_zthr_summary.py
    # parents[0]=core, parents[1]=src, parents[2]=tamcsismico
    return Path(__file__).resolve().parents[2]


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists() and path.is_file():
        return pd.read_csv(path)
    return None


def _ensure_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _quantile_band(df: pd.DataFrame, metric: str, q_low: float = 0.05, q_high: float = 0.95) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      event, zthr, p_low, p50, p_high
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["event", "zthr", "p_low", "p50", "p_high"])

    if not {"event", "zthr", metric}.issubset(df.columns):
        return pd.DataFrame(columns=["event", "zthr", "p_low", "p50", "p_high"])

    tmp = df[["event", "zthr", metric]].copy()
    tmp["zthr"] = _ensure_float(tmp["zthr"])
    tmp[metric] = _ensure_float(tmp[metric])
    tmp = tmp.dropna(subset=["event", "zthr", metric])

    qs = [q_low, 0.5, q_high]
    q = tmp.groupby(["event", "zthr"], dropna=False)[metric].quantile(qs)

    # q tiene MultiIndex (event, zthr, quantile)
    out = q.unstack(level=-1).reset_index()
    # columnas: event, zthr, 0.05, 0.5, 0.95
    # renombrar robusto:
    colmap = {}
    for c in out.columns:
        if isinstance(c, float) and abs(c - q_low) < 1e-12:
            colmap[c] = "p_low"
        elif isinstance(c, float) and abs(c - 0.5) < 1e-12:
            colmap[c] = "p50"
        elif isinstance(c, float) and abs(c - q_high) < 1e-12:
            colmap[c] = "p_high"
    out = out.rename(columns=colmap)

    # Si por lo que sea faltan columnas (p.ej., pocos datos), créalas
    for c in ["p_low", "p50", "p_high"]:
        if c not in out.columns:
            out[c] = np.nan

    return out[["event", "zthr", "p_low", "p50", "p_high"]].sort_values(["event", "zthr"])


def _pick_event(available: List[str], event: Optional[str], event_substr: Optional[str]) -> str:
    if event:
        if event in available:
            return event
        raise ValueError(f"Event '{event}' not found in available events (n={len(available)}).")

    if not event_substr:
        raise ValueError("Provide either --event (exact) or --event-substr (substring).")

    matches = [e for e in available if event_substr.lower() in e.lower()]
    if len(matches) == 0:
        examples = available[:8]
        raise ValueError(f"No event matches event_substr='{event_substr}'. Available examples: {examples} ...")
    if len(matches) > 1:
        # si hay varias coincidencias, elegimos la más “corta” (suele ser más específica)
        matches = sorted(matches, key=len)
    return matches[0]


def _read_real_from_c_robustness(base_dir: Path, zthr_list: List[int]) -> pd.DataFrame:
    """
    Lee C_robustness/zthr_*/real_multi_phase_summary.csv y concatena.
    Añade columna zthr si no existe (la infiere del nombre de carpeta).
    """
    rows = []
    for z in zthr_list:
        p = base_dir / "C_robustness" / f"zthr_{z}" / "real_multi_phase_summary.csv"
        df = _read_csv_if_exists(p)
        if df is None or df.empty:
            continue

        df = df.copy()
        if "zthr" not in df.columns:
            df["zthr"] = float(z)
        else:
            df["zthr"] = _ensure_float(df["zthr"]).fillna(float(z))

        # Normaliza nombres típicos
        if "event" not in df.columns:
            # algunos ficheros podrían usar "event_id"
            for alt in ["event_id", "name"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "event"})
                    break

        if "event" not in df.columns:
            continue

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


def _make_out_paths(base_dir: Path, out_dir: Optional[str], metric: str, zmin: int, zmax: int) -> Tuple[Path, Path]:
    if out_dir:
        od = Path(out_dir)
    else:
        od = base_dir / "analysis" / "figures"
    od.mkdir(parents=True, exist_ok=True)

    png = od / f"robustness_{metric}_zthr{zmin}_{zmax}.png"
    csv = od / f"robustness_{metric}_zthr{zmin}_{zmax}.csv"
    return png, csv


# -----------------------------
# Main plotting
# -----------------------------

def plot_robustness(
    real_df: pd.DataFrame,
    nullA: Optional[pd.DataFrame],
    nullB: Optional[pd.DataFrame],
    event_name: str,
    metric: str,
    zthr_min: int,
    zthr_max: int,
    out_png: Path,
    out_csv: Path,
) -> None:
    # --- real
    if real_df is None or real_df.empty:
        raise ValueError("No real robustness data loaded (C_robustness/zthr_*/real_multi_phase_summary.csv).")

    if metric not in real_df.columns:
        raise ValueError(f"Metric '{metric}' not present in real data. Available: {list(real_df.columns)}")

    real_e = real_df[real_df["event"] == event_name].copy()
    if real_e.empty:
        # list some
        avail = sorted(real_df["event"].dropna().unique().tolist())
        raise ValueError(f"Event '{event_name}' not found in real robustness data. Examples: {avail[:8]} ...")

    real_e["zthr"] = _ensure_float(real_e["zthr"])
    real_e[metric] = _ensure_float(real_e[metric])
    real_e = real_e.dropna(subset=["zthr", metric]).sort_values("zthr")

    # --- null bands
    qa = _quantile_band(nullA, metric) if nullA is not None else pd.DataFrame()
    qb = _quantile_band(nullB, metric) if nullB is not None else pd.DataFrame()

    qa_e = qa[qa["event"] == event_name].copy() if not qa.empty else pd.DataFrame()
    qb_e = qb[qb["event"] == event_name].copy() if not qb.empty else pd.DataFrame()

    # --- x range
    x_all = set(real_e["zthr"].tolist())
    if not qa_e.empty:
        x_all |= set(qa_e["zthr"].tolist())
    if not qb_e.empty:
        x_all |= set(qb_e["zthr"].tolist())

    # Si aun así solo hay 1 punto, no pasa nada: se dibuja, pero avisamos en stdout con el CSV final.
    x = sorted([v for v in x_all if np.isfinite(v)])
    if len(x) == 0:
        raise ValueError("No zthr values available after filtering.")

    # Guardamos CSV combinado (por si lo quieres inspeccionar)
    combined = pd.DataFrame({"zthr": x})
    # Real: puede haber múltiples filas por zthr; nos quedamos con la primera (o la media si hay varias)
    real_agg = real_e.groupby("zthr")[metric].mean().reset_index().rename(columns={metric: "real"})
    combined = combined.merge(real_agg, on="zthr", how="left")

    def _merge_band(dst: pd.DataFrame, band: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if band is None or band.empty:
            for c in [f"{prefix}_p_low", f"{prefix}_p50", f"{prefix}_p_high"]:
                dst[c] = np.nan
            return dst
        b = band[["zthr", "p_low", "p50", "p_high"]].copy()
        b = b.rename(columns={
            "p_low": f"{prefix}_p_low",
            "p50": f"{prefix}_p50",
            "p_high": f"{prefix}_p_high",
        })
        return dst.merge(b, on="zthr", how="left")

    combined = _merge_band(combined, qa_e, "nullA")
    combined = _merge_band(combined, qb_e, "nullB")
    combined.to_csv(out_csv, index=False)

    # --- plot
    plt.figure(figsize=(11.5, 6.5))

    # Null A
    if not qa_e.empty:
        xa = qa_e["zthr"].to_numpy()
        plt.fill_between(xa, qa_e["p_low"].to_numpy(), qa_e["p_high"].to_numpy(), alpha=0.2, label="Null A (p5–p95)")
        plt.plot(xa, qa_e["p50"].to_numpy(), linewidth=2.2, label="Null A (median)")

    # Null B
    if not qb_e.empty:
        xb = qb_e["zthr"].to_numpy()
        plt.fill_between(xb, qb_e["p_low"].to_numpy(), qb_e["p_high"].to_numpy(), alpha=0.2, label="Null B (p5–p95)")
        plt.plot(xb, qb_e["p50"].to_numpy(), linewidth=2.2, linestyle="--", label="Null B (median)")

    # Real event points
    plt.plot(real_e["zthr"].to_numpy(), real_e[metric].to_numpy(), marker="o", linewidth=2.2, label="Real event")

    plt.xlabel("z$_{thr}$")
    plt.ylabel(metric)
    plt.title(f"Robustness across thresholds: {event_name}\nmetric={metric}")
    plt.grid(True, alpha=0.25)

    # Mejor xlim: usa el rango solicitado, pero si no hay datos, ajusta a lo que haya
    xmin = zthr_min
    xmax = zthr_max
    plt.xlim(xmin, xmax)

    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Directorio base donde vive null_tests_13_14 (por defecto: <repo>/resultados/null_tests_13_14)")
    parser.add_argument("--event", type=str, default=None, help="Nombre exacto del evento")
    parser.add_argument("--event-substr", type=str, default=None, help="Substring para encontrar el evento (ej: 'Tohoku')")
    parser.add_argument("--metric", type=str, default="n_phases", help="Métrica a plotear (ej: n_phases, phase1_duration_h, ...)")
    parser.add_argument("--zthr-min", type=int, default=3)
    parser.add_argument("--zthr-max", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default=None, help="Directorio salida para PNG/CSV (por defecto: <base>/analysis/figures)")
    parser.add_argument("--nullA-csv", type=str, default=None, help="Ruta a nullA_simulations.csv (opcional)")
    parser.add_argument("--nullB-csv", type=str, default=None, help="Ruta a nullB_simulations.csv (opcional)")
    args = parser.parse_args()

    repo_root = _repo_root_from_this_file()
    base_dir = Path(args.base_dir) if args.base_dir else (repo_root / "resultados" / "null_tests_13_14")
    if not base_dir.exists():
        raise ValueError(f"base_dir not found: {base_dir}")

    zthr_list = list(range(args.zthr_min, args.zthr_max + 1))

    # Real (C_robustness)
    real_df = _read_real_from_c_robustness(base_dir, zthr_list)
    if real_df.empty:
        raise ValueError(f"No real data found under: {base_dir / 'C_robustness'} (zthr_{args.zthr_min}..zthr_{args.zthr_max})")

    if "event" not in real_df.columns:
        raise ValueError("Real robustness CSVs do not contain an 'event' column.")

    available_events = sorted(real_df["event"].dropna().unique().tolist())
    event_name = _pick_event(available_events, args.event, args.event_substr)

    # Null sims (opcionales)
    nullA_path = Path(args.nullA_csv) if args.nullA_csv else (base_dir / "nullA_simulations.csv")
    nullB_path = Path(args.nullB_csv) if args.nullB_csv else (base_dir / "nullB_simulations.csv")
    nullA = _read_csv_if_exists(nullA_path)
    nullB = _read_csv_if_exists(nullB_path)

    # Outputs
    out_png, out_csv = _make_out_paths(base_dir, args.out_dir, args.metric, args.zthr_min, args.zthr_max)

    plot_robustness(
        real_df=real_df,
        nullA=nullA,
        nullB=nullB,
        event_name=event_name,
        metric=args.metric,
        zthr_min=args.zthr_min,
        zthr_max=args.zthr_max,
        out_png=out_png,
        out_csv=out_csv,
    )

    print(f"[OK] Event:   {event_name}")
    print(f"[OK] Metric:  {args.metric}")
    print(f"[OK] PNG:     {out_png}")
    print(f"[OK] CSV:     {out_csv}")

    # Mini-aviso si solo hay un zthr
    c = pd.read_csv(out_csv)
    npts = int(c["real"].notna().sum())
    if npts <= 1:
        print("[WARN] Solo hay 0–1 puntos reales en el rango zthr. "
              "Eso significa que en C_robustness solo existe una salida para ese zthr, "
              "o que el evento no aparece en zthr_3/5/6.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
