#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[ES] Visualización + p-values empíricos para Null Tests (A/B) de Tests 13 y 14.
[EN] Visualization + empirical p-values for Null Tests (A/B) of Tests 13 and 14.

[ES] Qué hace:
 - Lee resultados reales (Test 13 + 14) y simulaciones nulas (Null A y Null B).
 - Detecta automáticamente métricas numéricas comunes entre real y null.
 - Calcula p-values empíricos:
    * one-sided (cola superior e inferior)
    * two-sided (dos colas por distancia al observado)
 - Genera histogramas (Null A y Null B) con línea vertical del observado.
 - Exporta CSVs con tablas de p-values y guarda PNGs.

[EN] What it does:
 - Loads real outputs (Test 13 + 14) and null simulations (Null A and Null B).
 - Auto-detects numeric metrics common to real and null.
 - Computes empirical p-values:
    * one-sided (upper / lower tail)
    * two-sided (two tails using distance to observed)
 - Produces histograms (Null A and Null B) with observed vertical line.
 - Writes p-value tables and saves PNGs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# [ES] Utilidades de paths / búsqueda de archivos
# [EN] Path helpers / file discovery
# ============================================================

DEFAULT_INDIR_REL = Path("resultados") / "null_tests_13_14"


def find_file_recursive(root: Path, filename: str) -> Optional[Path]:
    """
    [ES] Busca un archivo por nombre dentro de root (recursivo). Devuelve Path o None.
    [EN] Recursively search for a file by name under root. Returns Path or None.
    """
    hits = list(root.rglob(filename))
    return hits[0] if hits else None


def ensure_dir(p: Path) -> None:
    """
    [ES] Crea el directorio si no existe.
    [EN] Create directory if missing.
    """
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# [ES] Identificación de columnas / métricas
# [EN] Column / metric identification
# ============================================================

def guess_event_col(df: pd.DataFrame) -> Optional[str]:
    """
    [ES] Intenta adivinar la columna que identifica el evento.
    [EN] Tries to guess the event identifier column.
    """
    candidates = ["event", "evento", "ev", "name", "dataset"]
    for c in candidates:
        if c in df.columns:
            return c
    # heurística: columna tipo object con pocos valores únicos
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        # elige la que tenga menos únicos (pero >1)
        best = None
        best_u = None
        for c in obj_cols:
            u = df[c].nunique(dropna=True)
            if u > 1 and (best_u is None or u < best_u):
                best = c
                best_u = u
        return best
    return None


def numeric_common_metrics(real_df: pd.DataFrame, null_df: pd.DataFrame, event_col: str) -> List[str]:
    """
    [ES] Devuelve lista de columnas numéricas que existen en real y null.
    [EN] Returns numeric columns existing in both real and null.
    """
    # numéricas en ambos
    real_num = {c for c in real_df.columns if c != event_col and pd.api.types.is_numeric_dtype(real_df[c])}
    null_num = {c for c in null_df.columns if c != event_col and pd.api.types.is_numeric_dtype(null_df[c])}
    common = sorted(list(real_num.intersection(null_num)))
    return common


# ============================================================
# [ES] P-values empíricos
# [EN] Empirical p-values
# ============================================================

def empirical_pvals(null_vals: np.ndarray, observed: float) -> Dict[str, float]:
    """
    [ES] Calcula p-values empíricos:
         - p_upper: P(null >= obs)
         - p_lower: P(null <= obs)
         - p_two_sided: 2-colas usando distancia |x-obs| >= |obs-mean?|
           Aquí usamos la definición robusta:
              p_two = P(|null - obs| >= 0)?? no.
           Mejor: p_two = P(|null - obs| >= |obs - median(null)|)? tampoco.
           Usamos una definición estándar para comparación con observado:
              p_two = P(|null - median(null)| >= |obs - median(null)|)
         Con corrección +1/(N+1) para evitar p=0.

    [EN] Computes empirical p-values:
         - p_upper: P(null >= obs)
         - p_lower: P(null <= obs)
         - p_two_sided: two-sided using distance from the null median:
              p_two = P(|null - med(null)| >= |obs - med(null)|)
         With +1/(N+1) correction to avoid p=0.
    """
    x = null_vals[np.isfinite(null_vals)]
    n = x.size
    if n == 0 or not np.isfinite(observed):
        return {"p_upper": np.nan, "p_lower": np.nan, "p_two_sided": np.nan, "n": float(n)}

    med = np.nanmedian(x)
    dist_obs = abs(observed - med)
    dist_null = np.abs(x - med)

    # +1 correction
    p_upper = (np.sum(x >= observed) + 1.0) / (n + 1.0)
    p_lower = (np.sum(x <= observed) + 1.0) / (n + 1.0)
    p_two = (np.sum(dist_null >= dist_obs) + 1.0) / (n + 1.0)

    return {"p_upper": float(p_upper), "p_lower": float(p_lower), "p_two_sided": float(p_two), "n": float(n)}


# ============================================================
# [ES] Plotting
# [EN] Plotting
# ============================================================

def plot_hist_with_observed(
    null_vals: np.ndarray,
    observed: float,
    title: str,
    xlabel: str,
    out_png: Path,
    bins: int = 40,
) -> None:
    """
    [ES] Histograma de null con línea vertical para observado.
         Robusto ante:
           - pocos datos finitos (0/1)
           - rango degenerado (min==max) que rompe np.histogram con muchos bins
    [EN] Null histogram with observed vertical line.
         Robust to:
           - too few finite points (0/1)
           - degenerate range (min==max) which breaks np.histogram with many bins
    """
    x = np.asarray(null_vals, dtype=float)
    x = x[np.isfinite(x)]
    ensure_dir(out_png.parent)

    plt.figure()

    if x.size >= 2:
        xmin = float(np.min(x))
        xmax = float(np.max(x))

        # If range is (near) zero, fall back to 1 bin with a small epsilon range.
        if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or np.isclose(xmin, xmax):
            x0 = xmin if np.isfinite(xmin) else 0.0
            eps = 1e-12 if x0 == 0 else abs(x0) * 1e-6
            plt.hist(x, bins=1, range=(x0 - eps, x0 + eps))
        else:
            # Avoid asking for more bins than points (helps stability)
            nb = int(bins) if isinstance(bins, int) else 40
            nb = max(1, min(nb, int(x.size)))
            plt.hist(x, bins=nb)

    elif x.size == 1:
        # Single value: draw a tiny 1-bin histogram around it
        x0 = float(x[0])
        eps = 1e-12 if x0 == 0 else abs(x0) * 1e-6
        plt.hist(x, bins=1, range=(x0 - eps, x0 + eps))
    else:
        # No finite data: just annotate
        plt.text(0.5, 0.5, "No finite null data", ha="center", va="center", transform=plt.gca().transAxes)

    if np.isfinite(observed):
        plt.axvline(observed, linestyle="--")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia / Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ============================================================
# [ES] Carga de datos (real + null)
# [EN] Load data (real + null)
# [EN] Load data (real + null)
# ============================================================

def load_real_tables(indir: Path) -> pd.DataFrame:
    """
    [ES] Carga y combina:
        - real_multi_phase_summary.csv  (Test 13)
        - real_station_coverage_by_phase.csv (Test 14)
        en una sola tabla por evento.
    [EN] Loads and merges:
        - real_multi_phase_summary.csv  (Test 13)
        - real_station_coverage_by_phase.csv (Test 14)
        into a single per-event table.
    """
    f13 = find_file_recursive(indir, "real_multi_phase_summary.csv")
    f14 = find_file_recursive(indir, "real_station_coverage_by_phase.csv")

    if f13 is None:
        raise FileNotFoundError(f"[ES] No encuentro real_multi_phase_summary.csv en {indir}\n"
                                f"[EN] Cannot find real_multi_phase_summary.csv under {indir}")
    if f14 is None:
        raise FileNotFoundError(f"[ES] No encuentro real_station_coverage_by_phase.csv en {indir}\n"
                                f"[EN] Cannot find real_station_coverage_by_phase.csv under {indir}")

    df13 = pd.read_csv(f13)
    df14 = pd.read_csv(f14)

    ev13 = guess_event_col(df13)
    ev14 = guess_event_col(df14)
    if ev13 is None or ev14 is None:
        raise ValueError("[ES] No pude detectar columna de evento en real.\n"
                         "[EN] Could not detect event column in real tables.")

    # normaliza nombre col evento
    if ev13 != "event":
        df13 = df13.rename(columns={ev13: "event"})
    if ev14 != "event":
        df14 = df14.rename(columns={ev14: "event"})

    # merge outer para no perder nada
    df = pd.merge(df13, df14, on="event", how="outer", suffixes=("", "_cov"))
    return df


def load_null_table(indir: Path, filename: str) -> pd.DataFrame:
    """
    [ES] Carga nullA_simulations.csv o nullB_simulations.csv.
    [EN] Loads nullA_simulations.csv or nullB_simulations.csv.
    """
    f = find_file_recursive(indir, filename)
    if f is None:
        raise FileNotFoundError(f"[ES] No encuentro {filename} en {indir}\n"
                                f"[EN] Cannot find {filename} under {indir}")
    df = pd.read_csv(f)
    ev = guess_event_col(df)
    if ev is None:
        raise ValueError(f"[ES] No pude detectar columna de evento en {filename}.\n"
                         f"[EN] Could not detect event column in {filename}.")
    if ev != "event":
        df = df.rename(columns={ev: "event"})
    return df


# ============================================================
# [ES] Pipeline principal de análisis
# [EN] Main analysis pipeline
# ============================================================

def analyze(
    indir: Path,
    outdir: Path,
    bins: int,
    metrics: Optional[List[str]] = None,
    selected_events: Optional[List[str]] = None,
) -> None:
    """
    [ES] Ejecuta el análisis completo: p-values + PNGs.
    [EN] Runs full analysis: p-values + PNGs.
    """
    ensure_dir(outdir)
    figdir = outdir / "figures"
    ensure_dir(figdir)

    print(f"[ES] Leyendo datos desde: {indir}")
    print(f"[EN] Reading data from: {indir}")

    real = load_real_tables(indir)
    nullA = load_null_table(indir, "nullA_simulations.csv")
    nullB = load_null_table(indir, "nullB_simulations.csv")

    # filtra por eventos seleccionados (si se proporcionan)
    if selected_events:
        sel = set(selected_events)
        real = real[real["event"].astype(str).isin(sel)].copy()
        nullA = nullA[nullA["event"].astype(str).isin(sel)].copy()
        nullB = nullB[nullB["event"].astype(str).isin(sel)].copy()

        if real.empty:
            raise FileNotFoundError(
                "[ES] Ninguno de los eventos seleccionados aparece en real_multi_phase_summary.csv / real_station_coverage_by_phase.csv.\n"
                "[EN] None of the selected events are present in the real tables."
            )


    # métricas comunes
    commonA = numeric_common_metrics(real, nullA, "event")
    commonB = numeric_common_metrics(real, nullB, "event")
    common = sorted(list(set(commonA).intersection(commonB)))

    if metrics:
        # filtra a las pedidas por el usuario
        missing = [m for m in metrics if m not in common]
        if missing:
            print("[ES] Aviso: algunas métricas pedidas no están en ambos nulls + real:", missing)
            print("[EN] Warning: some requested metrics are not in both nulls + real:", missing)
        common = [m for m in metrics if m in common]

    if not common:
        raise RuntimeError(
            "[ES] No encuentro métricas numéricas comunes entre real, nullA y nullB.\n"
            "[EN] No common numeric metrics found across real, nullA and nullB."
        )

    print(f"[ES] Métricas analizadas ({len(common)}): {common}")
    print(f"[EN] Metrics analyzed ({len(common)}): {common}")

    # tabla larga detallada
    rows_detail: List[Dict[str, object]] = []

    # por evento
    events = sorted([e for e in real["event"].dropna().unique().tolist()])
    for ev in events:
        real_row = real[real["event"] == ev]
        if real_row.empty:
            continue
        real_row = real_row.iloc[0]

        a_ev = nullA[nullA["event"] == ev]
        b_ev = nullB[nullB["event"] == ev]

        for metric in common:
            obs = real_row.get(metric, np.nan)
            a_vals = a_ev[metric].to_numpy(dtype=float) if metric in a_ev.columns else np.array([], dtype=float)
            b_vals = b_ev[metric].to_numpy(dtype=float) if metric in b_ev.columns else np.array([], dtype=float)

            pa = empirical_pvals(a_vals, float(obs) if np.isfinite(obs) else np.nan)
            pb = empirical_pvals(b_vals, float(obs) if np.isfinite(obs) else np.nan)

            # guarda fila
            rows_detail.append({
                "event": ev,
                "metric": metric,
                "observed": float(obs) if np.isfinite(obs) else np.nan,
                "nullA_n": int(pa["n"]),
                "nullA_p_upper": pa["p_upper"],
                "nullA_p_lower": pa["p_lower"],
                "nullA_p_two_sided": pa["p_two_sided"],
                "nullB_n": int(pb["n"]),
                "nullB_p_upper": pb["p_upper"],
                "nullB_p_lower": pb["p_lower"],
                "nullB_p_two_sided": pb["p_two_sided"],
            })

            # plots
            safe_metric = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in metric)

            titleA = f"[Null A] {ev} — {metric}"
            titleB = f"[Null B] {ev} — {metric}"
            xlabel = f"{metric} (obs= {obs})"

            outA = figdir / f"{ev}__{safe_metric}__nullA.png"
            outB = figdir / f"{ev}__{safe_metric}__nullB.png"

            plot_hist_with_observed(a_vals, float(obs) if np.isfinite(obs) else np.nan, titleA, xlabel, outA, bins=bins)
            plot_hist_with_observed(b_vals, float(obs) if np.isfinite(obs) else np.nan, titleB, xlabel, outB, bins=bins)

    df_detail = pd.DataFrame(rows_detail)

    # resumen: para cada evento, un “score” simple
    # (min p_two_sided entre métricas, por null A/B)
    def min_p(series: pd.Series) -> float:
        s = series.dropna()
        return float(s.min()) if not s.empty else np.nan

    df_summary = (
        df_detail
        .groupby("event", as_index=False)
        .agg(
            metrics_count=("metric", "count"),
            min_nullA_p_two_sided=("nullA_p_two_sided", min_p),
            min_nullB_p_two_sided=("nullB_p_two_sided", min_p),
            min_nullA_p_upper=("nullA_p_upper", min_p),
            min_nullB_p_upper=("nullB_p_upper", min_p),
        )
        .sort_values("event")
    )

    # write outputs
    out_detail = outdir / "pvalues_detailed.csv"
    out_summary = outdir / "pvalues_summary.csv"
    df_detail.to_csv(out_detail, index=False)
    df_summary.to_csv(out_summary, index=False)

    print("\n[ES] OK. Archivos generados:")
    print("[EN] OK. Generated files:")
    print(f"  - {out_detail}")
    print(f"  - {out_summary}")
    print(f"  - PNGs en / PNGs in: {figdir}")


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    [ES] CLI robusta.
    [EN] Robust CLI.
    """
    p = argparse.ArgumentParser(
        description="[ES] Null viz + p-values (Tests 13/14) | [EN] Null viz + p-values (Tests 13/14)"
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="[ES] Root del repo (si no se da, usa CWD). "
             "[EN] Repo root (if omitted, uses CWD).",
    )
    p.add_argument(
        "--indir",
        type=str,
        default=None,
        help="[ES] Carpeta donde están los CSV (default: resultados/null_tests_13_14). "
             "[EN] Folder containing CSVs (default: resultados/null_tests_13_14).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="[ES] Carpeta de salida (default: <indir>/analysis). "
             "[EN] Output folder (default: <indir>/analysis).",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=40,
        help="[ES] Bins para histogramas. [EN] Histogram bins.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="[ES] Lista opcional de métricas a analizar (si no, auto-detect). "
             "[EN] Optional list of metrics (otherwise auto-detect).",
    )
    p.add_argument(
        "events",
        nargs="*",
        help="[ES] Eventos a incluir (nombres de carpeta en resultados/). Si se omite, usa todos. "
             "[EN] Events to include (folder names in resultados/). If omitted, uses all.",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()

    indir = Path(args.indir).resolve() if args.indir else (root / DEFAULT_INDIR_REL).resolve()
    if not indir.exists():
        # fallback: buscar carpeta por nombre en root
        cand = find_file_recursive(root, "real_multi_phase_summary.csv")
        if cand is not None:
            indir = cand.parent
        else:
            raise FileNotFoundError(
                f"[ES] No encuentro indir: {indir}\n"
                f"[EN] Cannot find indir: {indir}\n"
                f"[ES] Pista: usa --indir para apuntar a resultados/null_tests_13_14\n"
                f"[EN] Tip: use --indir to point to resultados/null_tests_13_14"
            )

    outdir = Path(args.outdir).resolve() if args.outdir else (indir / "analysis").resolve()

    analyze(
        indir=indir,
        outdir=outdir,
        bins=int(args.bins),
        metrics=args.metrics,
        selected_events=args.events,
    )


if __name__ == "__main__":
    main()
