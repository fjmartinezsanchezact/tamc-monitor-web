#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
18_extract_final_pvalues.py

- Lee resultados por evento de NullA y RobC y construye tabla final de p-values + sigmas.
- Escribe:
    resultados/null_tests_13_14/analysis/paper/final_pvalues_table.csv
    resultados/null_tests_13_14/analysis/paper/final_pvalues_table.tex
- NUEVO:
    - imprime la tabla por pantalla
    - genera un PNG con la tabla:
      resultados/null_tests_13_14/analysis/paper/final_pvalues_table.png

Todo con rutas relativas (detecta root del repo automáticamente).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# IMPORTANTE: backend sin GUI (evita errores Tkinter)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utils: repo root
# -------------------------
def find_repo_root(start: Path) -> Path:
    """
    Sube directorios hasta encontrar la raíz del repo.
    Criterios: existe carpeta 'src' y 'resultados' (o al menos 'src').
    """
    p = start.resolve()
    for _ in range(20):
        if (p / "src").exists() and ((p / "resultados").exists() or (p / "data").exists()):
            return p
        if (p / "src").exists():
            # si solo existe src, también lo aceptamos como root
            return p
        if p.parent == p:
            break
        p = p.parent
    # fallback: carpeta actual
    return start.resolve()


# -------------------------
# Stats
# -------------------------
def p_to_sigma_two_sided(p: float) -> float:
    """
    Aproxima sigma equivalente (two-sided) usando inverse-erf.
    sigma = sqrt(2) * erfcinv(p)  (two-sided)
    Aquí implementamos con scipy si existiera, pero sin depender de scipy:
    Usamos aproximación por inverse normal (Beasley-Springer/Moro simplificada).
    Para p muy pequeño -> sigma grande. Clampeamos.
    """
    if p is None or (isinstance(p, float) and (math.isnan(p))):
        return float("nan")

    p = float(p)
    p = max(min(p, 1.0), 1e-300)  # clamp

    # Convert two-sided p to one-sided tail prob
    q = p / 2.0

    # Inversa aproximada de la CDF normal estándar:
    # https://www.johndcook.com/blog/normal_cdf_inverse/
    # (aprox racional de Peter J. Acklam)
    def ndtri(u: float) -> float:
        # Coefs Acklam
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]

        plow = 0.02425
        phigh = 1 - plow

        if u < plow:
            r = math.sqrt(-2 * math.log(u))
            return (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) / \
                   ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1)
        if u > phigh:
            r = math.sqrt(-2 * math.log(1 - u))
            return -(((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) / \
                    ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1)

        r = u - 0.5
        s = r * r
        return (((((a[0]*s + a[1])*s + a[2])*s + a[3])*s + a[4])*s + a[5]) * r / \
               (((((b[0]*s + b[1])*s + b[2])*s + b[3])*s + b[4])*s + 1)

    # sigma (two-sided): z tal que P(|Z|>=z)=p => P(Z>=z)=p/2 => z = ndtri(1 - p/2)
    sigma = ndtri(1.0 - q)
    return float(sigma)


def empirical_p_ge(null_values: np.ndarray, real_value: float) -> float:
    """
    p = (#{null >= real} + 1) / (N + 1)  (one-sided, conservative)
    Maneja NaNs.
    """
    if real_value is None or (isinstance(real_value, float) and np.isnan(real_value)):
        return float("nan")

    x = np.asarray(null_values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    ge = np.sum(x >= real_value)
    p = (ge + 1.0) / (x.size + 1.0)
    return float(p)


# -------------------------
# Plot table PNG
# -------------------------
def save_table_png(df: pd.DataFrame, out_png: Path, title: str = "") -> None:
    """
    Renderiza un DataFrame como imagen PNG (tabla).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Ajuste de tamaño dinámico
    nrows = len(df)
    ncols = len(df.columns)
    # heurística: ancho por columnas, alto por filas
    fig_w = max(10, 1.2 * ncols)
    fig_h = max(3, 0.35 * (nrows + 3))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Formateo ligero (strings ya vienen bien)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    if title:
        ax.set_title(title, fontsize=12, pad=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------

# -------------------------
# Args + autodetect
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=None,
        help="Ruta a la raíz del repo (tamcsismico). Si no se da, se autodetecta.",
    )

    # Compatible con el pipeline: eventos posicionales (igual que 13/14/15/17)
    ap.add_argument(
        "events",
        nargs="*",
        help="Eventos a incluir. Si no se dan, se autodetectan desde los CSVs en analysis_dir.",
    )

    # Compatibilidad hacia atrás: --events (legacy)
    ap.add_argument(
        "--events",
        dest="events_opt",
        nargs="*",
        default=None,
        help="(Legacy) Lista de eventos. Si se dan también posicionales, se priorizan los posicionales.",
    )

    ap.add_argument(
        "--analysis_dir",
        default=None,
        help="Ruta (relativa o absoluta) a resultados/null_tests_13_14/analysis. Si no se da, se usa <root>/resultados/null_tests_13_14/analysis.",
    )
    ap.add_argument("--paper_subdir", default="paper")
    ap.add_argument("--out_csv", default="final_pvalues_table.csv")
    ap.add_argument("--out_tex", default="final_pvalues_table.tex")
    ap.add_argument("--out_png", default="final_pvalues_table.png")
    ap.add_argument("--quiet", action="store_true", help="No imprime por pantalla.")
    ap.add_argument("--png", action="store_true", help="Genera PNG con la tabla.")
    ap.add_argument("--title", default="Final p-values (Tests 13–14): NullA & RobustC")
    ap.add_argument("--print", dest="print", action="store_true", help="Imprime la tabla final por pantalla.")
    return ap.parse_args()


def autodetect_events_from_analysis(analysis_dir: Path) -> List[str]:
    """
    Autodetecta eventos a partir de outputs ya generados en analysis_dir.
    Busca (por prioridad): *_real_metrics.csv, *_nullA_simulations.csv, *_robC_simulations.csv
    """
    if not analysis_dir.exists():
        return []

    suffixes = [
        "_real_metrics.csv",
        "_nullA_simulations.csv",
        "_robC_simulations.csv",
    ]

    found = set()
    for suf in suffixes:
        for fp in analysis_dir.glob(f"*{suf}"):
            name = fp.name
            if name.endswith(suf):
                found.add(name[: -len(suf)])
    return sorted(found)


def resolve_root_and_analysis_dir(args: argparse.Namespace) -> Path:
    here = Path(__file__).resolve()
    if args.root:
        root = Path(args.root).resolve()
    else:
        root = find_repo_root(here.parent)
    return root


def main() -> None:
    args = parse_args()

    root = resolve_root_and_analysis_dir(args)

    if args.analysis_dir is None:
        analysis_dir = root / "resultados" / "null_tests_13_14" / "analysis"
    else:
        analysis_dir = Path(args.analysis_dir)
        if not analysis_dir.is_absolute():
            analysis_dir = (root / analysis_dir).resolve()
    # Resolver eventos: posicionales > --events legacy > autodetect
    events = list(args.events) if args.events else (list(args.events_opt) if args.events_opt else autodetect_events_from_analysis(analysis_dir))
    if not events:
        raise FileNotFoundError(
            "[ES] No pude determinar eventos (ni por argumentos ni por autodetección) en:\n"
            f"  {analysis_dir}\n"
            "[EN] Cannot determine events (neither via args nor autodetection) in:\n"
            f"  {analysis_dir}"
        )

    paper_dir = analysis_dir / args.paper_subdir
    paper_dir.mkdir(parents=True, exist_ok=True)

    out_csv = paper_dir / args.out_csv
    out_tex = paper_dir / args.out_tex
    out_png = paper_dir / args.out_png

    # Intentamos leer "real_multi_phase_summary.csv" si existe;
    # si no, usamos *_real_metrics.csv por evento si existen.
    real_summary_path = analysis_dir / "real_multi_phase_summary.csv"
    real_summary = None
    if real_summary_path.exists():
        real_summary = pd.read_csv(real_summary_path)

    metrics = ["n_phases", "phase1_duration_h", "gap_1_2_h", "phase1_fraction_active", "phase2_fraction_active"]

    rows: List[Dict[str, object]] = []

    for ev in events:
        print(f"[EVENT] {ev}")

        # Real metrics
        real_metrics: Dict[str, float] = {}

        if real_summary is not None and "event" in real_summary.columns:
            sub = real_summary[real_summary["event"] == ev]
            if len(sub) > 0:
                for m in metrics:
                    if m in sub.columns:
                        real_metrics[m] = float(sub.iloc[0][m])
        else:
            # fallback: por evento
            ev_real_path = analysis_dir / f"{ev}_real_metrics.csv"
            if ev_real_path.exists():
                tmp = pd.read_csv(ev_real_path)
                # acepta formato wide o long
                if "metric" in tmp.columns and "value" in tmp.columns:
                    for _, r in tmp.iterrows():
                        if r["metric"] in metrics:
                            real_metrics[str(r["metric"])] = float(r["value"])
                else:
                    # wide (1 fila)
                    for m in metrics:
                        if m in tmp.columns:
                            real_metrics[m] = float(tmp.iloc[0][m])

        # NullA sims y RobC sims
        nullA_path = analysis_dir / f"{ev}_nullA_simulations.csv"
        robC_path = analysis_dir / f"{ev}_robC_simulations.csv"

        if not nullA_path.exists():
            # soporte legacy
            legacy = analysis_dir / "paper" / f"{ev}_nullA_simulations.csv"
            if legacy.exists():
                nullA_path = legacy
        if not robC_path.exists():
            legacy = analysis_dir / "paper" / f"{ev}_robC_simulations.csv"
            if legacy.exists():
                robC_path = legacy

        nullA = pd.read_csv(nullA_path) if nullA_path.exists() else pd.DataFrame()
        robC = pd.read_csv(robC_path) if robC_path.exists() else pd.DataFrame()

        N_nullA = int(len(nullA)) if len(nullA) else 0
        N_robC = int(len(robC)) if len(robC) else 0

        for m in metrics:
            rv = real_metrics.get(m, float("nan"))

            pA = float("nan")
            pC = float("nan")
            if len(nullA) and m in nullA.columns:
                pA = empirical_p_ge(nullA[m].values, rv)
            if len(robC) and m in robC.columns:
                pC = empirical_p_ge(robC[m].values, rv)

            rows.append({
                "event": ev,
                "metric": m,
                "real_value": rv,
                "p_nullA": pA,
                "sigma_nullA": p_to_sigma_two_sided(pA) if not (isinstance(pA, float) and np.isnan(pA)) else float("nan"),
                "p_robC": pC,
                "sigma_robC": p_to_sigma_two_sided(pC) if not (isinstance(pC, float) and np.isnan(pC)) else float("nan"),
                "N_nullA": N_nullA,
                "N_robC": N_robC,
            })

    df = pd.DataFrame(rows)

    # Formato "paper friendly"
    def fmt_p(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.2e}"

    def fmt_sigma(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.2f}"

    df_out = df.copy()
    df_out["p_nullA"] = df_out["p_nullA"].apply(fmt_p)
    df_out["p_robC"] = df_out["p_robC"].apply(fmt_p)
    df_out["sigma_nullA"] = df_out["sigma_nullA"].apply(fmt_sigma)
    df_out["sigma_robC"] = df_out["sigma_robC"].apply(fmt_sigma)

    # CSV
    df_out.to_csv(out_csv, index=False)

    # LaTeX (booktabs)
    tex = df_out.to_latex(index=False, escape=False)
    out_tex.write_text(tex, encoding="utf-8")

    print("[OK] Written:")
    print(str(out_csv))
    print(str(out_tex))

    # Mostrar por pantalla
    if getattr(args, 'print', False):
        print("\n===== FINAL P-VALUES TABLE (Tests 13–14) =====")
        # impresión completa sin truncar
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(df_out.to_string(index=False))
        print("=============================================\n")

    # PNG
    if args.png:
        # Para el png, usamos df_out pero limitamos decimales y evitamos columnas muy largas
        save_table_png(df_out, out_png, title=args.title)
        print("[OK] PNG:", str(out_png))


if __name__ == "__main__":
    main()
