#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[ES] Genera tablas finales (paper-ready) a partir de pvalues_detailed.csv y/o pvalues_summary.csv
     usando SOLO rutas relativas (detecta el root del repo automáticamente).
[EN] Generates final paper-ready tables from pvalues_detailed.csv and/or pvalues_summary.csv
     using ONLY relative paths (auto-detects repo root).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# [ES] Detección automática del root del repo
# [EN] Automatic repo-root detection
# ============================================================
def find_repo_root(start: Path) -> Path:
    """
    [ES] Sube carpetas hasta encontrar la estructura típica del repo:
         - src/
         - resultados/
         - data/   (opcional pero común)
    [EN] Walks upwards until it finds typical repo structure:
         - src/
         - resultados/
         - data/   (optional but common)
    """
    cur = start.resolve()
    for _ in range(12):  # suficiente para no liar en rutas raras
        if (cur / "src").exists() and (cur / "resultados").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        "[ES] No pude detectar el root del repo. Ejecuta este script dentro de tamcsismico/.\n"
        "[EN] Could not detect repo root. Run this script inside tamcsismico/."
    )


# ============================================================
# [ES] Helpers de formato / [EN] Formatting helpers
# ============================================================
def fmt_p(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if x < 0.0001:
        return f"{x:.1e}"
    if x < 0.01:
        return f"{x:.4f}"
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.2f}"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if abs(x) < 0.01 and x != 0:
        return f"{x:.2e}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    [ES] Benjamini–Hochberg FDR q-values.
    [EN] Benjamini–Hochberg FDR q-values.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    idx = np.where(np.isfinite(p))[0]
    n = idx.size
    if n == 0:
        return out

    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[idx[order]] = q
    return out


def to_latex_table(df: pd.DataFrame, tex_path: Path, caption: str, label: str) -> None:
    """
    [ES] Exporta LaTeX en longtable.
    [EN] Export LaTeX as longtable.
    """
    tex = df.to_latex(
        index=False,
        escape=True,
        longtable=True,
        caption=caption,
        label=label,
    )
    tex_path.write_text(tex, encoding="utf-8")


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "[ES] Crea tablas LaTeX/CSV de p-values para el paper (Tests 13–14).\n"
            "[EN] Create LaTeX/CSV p-value tables for the paper (Tests 13–14)."
        )
    )
    ap.add_argument(
        "--indir",
        type=str,
        default=None,
        help=(
            "[ES] Carpeta analysis relativa al repo (default: resultados/null_tests_13_14/analysis)\n"
            "[EN] analysis folder relative to repo (default: resultados/null_tests_13_14/analysis)"
        ),
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=6,
        help="[ES] Top-K métricas por evento | [EN] Top-K metrics per event",
    )
    ap.add_argument(
        "--do_fdr",
        action="store_true",
        help="[ES] Añade q-values (BH-FDR) | [EN] Add q-values (BH-FDR)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # [ES] Detectamos root automáticamente desde la ubicación del script
    # [EN] Auto-detect repo root from script location
    this_file = Path(__file__).resolve()
    repo_root = find_repo_root(this_file.parent)

    # [ES] input dir relativo al repo
    # [EN] indir relative to repo
    indir = (repo_root / args.indir).resolve() if args.indir else (repo_root / "resultados" / "null_tests_13_14" / "analysis")

    f_sum = indir / "pvalues_summary.csv"
    f_det = indir / "pvalues_detailed.csv"

    if not f_sum.exists() and not f_det.exists():
        raise FileNotFoundError(
            f"[ES] No encuentro pvalues_summary.csv ni pvalues_detailed.csv en:\n  {indir}\n"
            f"[EN] Cannot find pvalues_summary.csv nor pvalues_detailed.csv in:\n  {indir}"
        )

    out_csv = indir / "paper_table_pvalues.csv"
    out_tex = indir / "paper_table_pvalues.tex"
    out_tex_top = indir / "paper_table_topmetrics.tex"

    # ============================================================
    # TABLA PRINCIPAL / MAIN TABLE (summary)
    # ============================================================
    if f_sum.exists():
        df = pd.read_csv(f_sum)

        keep = [c for c in [
            "event",
            "metrics_count",
            "min_nullA_p_two_sided",
            "min_nullB_p_two_sided",
            "min_nullA_p_upper",
            "min_nullB_p_upper",
        ] if c in df.columns]

        df = df[keep].copy()

        rename = {
            "event": "Evento / Event",
            "metrics_count": "Métricas / Metrics",
            "min_nullA_p_two_sided": "min p (Null A, 2-colas) / min p (Null A, two-sided)",
            "min_nullB_p_two_sided": "min p (Null B, 2-colas) / min p (Null B, two-sided)",
            "min_nullA_p_upper": "min p (Null A, cola sup.) / min p (Null A, upper)",
            "min_nullB_p_upper": "min p (Null B, cola sup.) / min p (Null B, upper)",
        }
        df = df.rename(columns=rename)

        for col in df.columns:
            if "min p" in col:
                df[col] = df[col].apply(fmt_p)

        df.to_csv(out_csv, index=False, encoding="utf-8")

        to_latex_table(
            df,
            out_tex,
            caption=(
                "Tabla resumen de significancia empírica (p-values) comparando resultados reales de los Tests 13–14 "
                "contra modelos nulos (Null A: time-shuffle; Null B: t0 aleatorio). "
                "Summary table of empirical significance (p-values) comparing real results of Tests 13–14 "
                "against null models (Null A: time-shuffle; Null B: random t0)."
            ),
            label="tab:null_pvalues_summary",
        )

        print(f"[OK] {out_csv.relative_to(repo_root)}")
        print(f"[OK] {out_tex.relative_to(repo_root)}")

    # ============================================================
    # TABLA SECUNDARIA / SECOND TABLE (top metrics)
    # ============================================================
    if f_det.exists():
        det = pd.read_csv(f_det)

        for c in ["event", "metric", "observed"]:
            if c not in det.columns:
                raise ValueError(f"[ES] Falta columna '{c}' en pvalues_detailed.csv | [EN] Missing '{c}'")

        if args.do_fdr and "nullB_p_two_sided" in det.columns:
            det["nullB_q_two_sided_FDR"] = bh_fdr(det["nullB_p_two_sided"].to_numpy())

        score_col = None
        for cand in ["nullB_p_two_sided", "nullA_p_two_sided", "nullB_p_upper", "nullA_p_upper"]:
            if cand in det.columns:
                score_col = cand
                break
        if score_col is None:
            raise ValueError("[ES] No encuentro columnas de p-value en detailed | [EN] No p-value columns found")

        det_rank = det.sort_values(["event", score_col], ascending=[True, True]).copy()
        top = det_rank.groupby("event", as_index=False).head(args.topk).copy()

        cols = ["event", "metric", "observed"]
        for cand in ["nullA_p_two_sided", "nullB_p_two_sided", "nullA_p_upper", "nullB_p_upper", "nullB_q_two_sided_FDR"]:
            if cand in top.columns:
                cols.append(cand)

        top = top[cols].copy().rename(columns={
            "event": "Evento / Event",
            "metric": "Métrica / Metric",
            "observed": "Observado / Observed",
            "nullA_p_two_sided": "p (Null A, 2-colas) / p (Null A, two-sided)",
            "nullB_p_two_sided": "p (Null B, 2-colas) / p (Null B, two-sided)",
            "nullA_p_upper": "p (Null A, cola sup.) / p (Null A, upper)",
            "nullB_p_upper": "p (Null B, cola sup.) / p (Null B, upper)",
            "nullB_q_two_sided_FDR": "q FDR (Null B, 2-colas) / q FDR (Null B, two-sided)",
        })

        top["Observado / Observed"] = top["Observado / Observed"].apply(lambda x: fmt_num(x))
        for c in top.columns:
            if c.startswith("p (") or c.startswith("q FDR"):
                top[c] = top[c].apply(fmt_p)

        to_latex_table(
            top,
            out_tex_top,
            caption=(
                f"Top-{args.topk} métricas por evento con mayor evidencia frente a modelos nulos "
                f"(ordenado por {score_col}). "
                f"Top-{args.topk} metrics per event with strongest evidence against null models "
                f"(sorted by {score_col})."
            ),
            label="tab:null_pvalues_topmetrics",
        )

        print(f"[OK] {out_tex_top.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
