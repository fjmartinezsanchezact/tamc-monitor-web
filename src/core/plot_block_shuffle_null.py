
# plot_block_shuffle_null.py
# -*- coding: utf-8 -*-
"""
Plot del null test 09 (block shuffle) a partir de los CSV que ya genera 09_null_block_shuffle.py

Entrada:
  - block_shuffle_null_distribution.csv  (columna: stat_null)
  - (opcional) block_shuffle_summary.csv (1 fila con stat_event, p_empirical, stat, zthr, etc.)

Salida:
  - PNG en el mismo directorio del CSV (por defecto)

Uso (Windows):
  python plot_block_shuffle_null.py --csv "...\block_shuffle_null_distribution.csv"
  python plot_block_shuffle_null.py --csv "...\block_shuffle_null_distribution.csv" --out "...\null_blockshuffle.png"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _infer_summary_path(dist_csv: Path) -> Path:
    # Mismo directorio, nombre estándar
    cand = dist_csv.parent / "block_shuffle_summary.csv"
    return cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta a block_shuffle_null_distribution.csv")
    ap.add_argument("--summary", default=None, help="Ruta a block_shuffle_summary.csv (opcional)")
    ap.add_argument("--out", default=None, help="Ruta PNG de salida (opcional)")
    ap.add_argument("--bins", type=int, default=40, help="Bins del histograma (default=40)")
    ap.add_argument("--title", default=None, help="Título (opcional)")
    ap.add_argument("--no-grid", action="store_true", help="Desactiva grid")
    args = ap.parse_args()

    dist_csv = Path(args.csv).expanduser()
    if not dist_csv.exists():
        raise FileNotFoundError(f"No existe: {dist_csv}")

    df = pd.read_csv(dist_csv)
    if "stat_null" not in df.columns:
        raise ValueError(f"CSV no tiene columna 'stat_null'. Columnas: {list(df.columns)}")

    x = pd.to_numeric(df["stat_null"], errors="coerce").dropna().values
    if x.size == 0:
        raise ValueError("No hay valores numéricos en stat_null.")

    # Summary (opcional pero recomendado)
    summary_path = Path(args.summary).expanduser() if args.summary else _infer_summary_path(dist_csv)
    summary = None
    if summary_path and summary_path.exists():
        try:
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
        except Exception:
            summary = None

    # Estadísticos
    q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])

    stat = None
    zthr = np.nan
    event = None
    stat_event = np.nan
    p_emp = np.nan
    block_size = np.nan
    n_shuffles = x.size

    if summary:
        stat = summary.get("stat", None)
        zthr = _safe_float(summary.get("zthr", np.nan))
        event = summary.get("event", None)
        stat_event = _safe_float(summary.get("stat_event", np.nan))
        p_emp = _safe_float(summary.get("p_empirical", np.nan))
        block_size = _safe_float(summary.get("block_size", np.nan))
        n_shuffles = int(_safe_float(summary.get("n_shuffles", x.size), x.size))

        # Si summary trae cuantiles, respétalos
        q05 = _safe_float(summary.get("null_q05", q05), q05)
        q50 = _safe_float(summary.get("null_q50", q50), q50)
        q95 = _safe_float(summary.get("null_q95", q95), q95)

    # Output
    if args.out:
        out_png = Path(args.out).expanduser()
    else:
        # Nombre automático
        base = "block_shuffle_null"
        if stat:
            base += f"_{stat}"
        out_png = dist_csv.parent / f"{base}.png"

    # Plot
    plt.figure(figsize=(9, 5.2))
    plt.hist(x, bins=args.bins)

    # Líneas de cuantiles
    plt.axvline(q05, linestyle="--", linewidth=2, label="Null q05")
    plt.axvline(q50, linestyle="-", linewidth=2, label="Null median")
    plt.axvline(q95, linestyle="--", linewidth=2, label="Null q95")

    # Real event
    if np.isfinite(stat_event):
        plt.axvline(stat_event, linestyle="-.", linewidth=3, label="Real event")

    xlabel = "stat_null"
    if stat:
        xlabel = f"{stat} (null)"
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    # Título
    if args.title:
        title = args.title
    else:
        title_parts = ["Block-shuffle null distribution"]
        if event:
            title_parts.append(str(event))
        if stat and np.isfinite(zthr):
            title_parts.append(f"{stat} (zthr={zthr:g})")
        elif stat:
            title_parts.append(str(stat))
        title = " — ".join(title_parts)

    plt.title(title)

    # Texto resumen
    txt_lines = [
        f"N={int(x.size)} (expected {n_shuffles})",
        f"q05={q05:.3g}, median={q50:.3g}, q95={q95:.3g}",
    ]
    if np.isfinite(stat_event):
        txt_lines.append(f"real={stat_event:.3g}")
    if np.isfinite(p_emp):
        txt_lines.append(f"p_emp={p_emp:.4g}")
    if np.isfinite(block_size):
        txt_lines.append(f"block_size={int(block_size)}")

    plt.gca().text(
        0.02, 0.98,
        "\n".join(txt_lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    if not args.no_grid:
        plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] PNG: {out_png}")
    if summary_path and summary_path.exists():
        print(f"[OK] SUMMARY USED: {summary_path}")
    else:
        print("[WARN] No summary CSV found; plotted only stat_null distribution.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
