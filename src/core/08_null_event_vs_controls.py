#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_null_event_vs_controls.py

Null test empírico EVENTO vs DÍAS CONTROL (layout "auditable"):

resultados/
├── <EVENT>/
│   └── metrics/tamc_24h_metrics_allstations.csv
├── <EVENT>_control_day_1/
│   └── metrics/tamc_24h_metrics_allstations.csv
├── <EVENT>_control_day_2/
│   └── metrics/tamc_24h_metrics_allstations.csv
└── ...
└── <EVENT>_nulltest/        <-- SALIDA (se crea automáticamente)
    ├── event_vs_controls_summary.csv
    ├── event_vs_controls_pvalues.txt
    └── event_vs_controls.png

Qué calcula (para EVENT y cada CONTROL):
- zmax_abs = max(|z|)
- N(|z|>=3), N(|z|>=4), N(|z|>=5)

p-valor empírico (con +1 smoothing conservador):
p = ( #{control >= evento} + 1 ) / (N_controls + 1)

Uso:
  python 11_null_event_vs_controls.py maule2010
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_THRESHOLDS = [3.0, 4.0, 5.0]


def project_root() -> Path:
    """
    Devuelve la raíz del repo asumiendo que este archivo está en:
    <root>/src/core/este_script.py  -> parents[2] = <root>
    """
    return Path(__file__).resolve().parents[2]


def find_metric_csv(folder: Path) -> Path:
    p = folder / "metrics" / "tamc_24h_metrics_allstations.csv"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    return p


def compute_stats_from_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    # Columna z
    if "zscore" in df.columns:
        z = df["zscore"].astype(float).to_numpy()
    elif "z_score" in df.columns:
        z = df["z_score"].astype(float).to_numpy()
    else:
        raise KeyError(
            f"{csv_path}: no encuentro columna zscore/z_score. "
            f"Columnas: {list(df.columns)}"
        )

    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError(f"{csv_path}: serie z vacía o no numérica")

    zabs = np.abs(z)

    out = {
        "zmax_abs": float(np.max(zabs)),
        "n_points": int(len(zabs)),
    }
    for th in Z_THRESHOLDS:
        out[f"count_abs_ge_{th:g}"] = int(np.sum(zabs >= th))
    return out


def empirical_pvalue(controls: np.ndarray, event_value: float) -> float:
    """
    p empírico conservador con smoothing +1:
    p = (#{controls >= event} + 1) / (N + 1)
    """
    controls = np.asarray(controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    n = len(controls)
    if n == 0:
        return float("nan")
    return float((np.sum(controls >= event_value) + 1) / (n + 1))


def main(event: str, results_dir: str = "resultados") -> None:
    root = project_root()
    resultados = root / results_dir

    # --------------------
    # Carpeta del evento
    # --------------------
    event_dir = resultados / event
    if not event_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de evento: {event_dir}")

    # --------------------
    # Carpetas control
    # --------------------
    # Convenciones soportadas para controles:
    #   1) "<EVENT>_control..." (por ejemplo: tohoku2011_control_2010-04-16)
    #   2) "control_<EVENT>_..." (por ejemplo: control_tohoku2011_2010-04-16)
    prefixes = [f"{event}_control", f"control_{event}"]
    control_dirs = sorted(
        [
            d
            for d in resultados.iterdir()
            if d.is_dir() and any(d.name.startswith(pfx) for pfx in prefixes)
        ],
        key=lambda p: p.name,
    )
    if len(control_dirs) == 0:
        raise RuntimeError(
            f"No encontré controles con prefijo '{prefix}' en:\n{resultados}\n\n"
            f"Ejemplos válidos:\n"
            f"  {event}_control_day_1\n"
            f"  {event}_control_day_2\n"
            f"  {event}_control_2010-02-10\n"
        )

    print(f"[OK] Evento: {event_dir.name}")
    print(f"[OK] Controles encontrados: {len(control_dirs)}")

    # --------------------
    # Estadísticos EVENTO
    # --------------------
    event_csv = find_metric_csv(event_dir)
    ev_stats = compute_stats_from_metrics(event_csv)
    rows = [{"group": "EVENT", "name": event, **ev_stats}]

    # --------------------
    # Estadísticos CONTROLES
    # --------------------
    for cd in control_dirs:
        try:
            c_csv = find_metric_csv(cd)
            c_stats = compute_stats_from_metrics(c_csv)
            rows.append({"group": "CONTROL", "name": cd.name, **c_stats})
        except Exception as e:
            print(f"[WARN] Saltando control {cd.name}: {e}")

    df = pd.DataFrame(rows)
    controls = df[df["group"] == "CONTROL"].copy()

    if len(controls) < 3:
        print(f"[WARN] Pocos controles válidos ({len(controls)}). Ideal >= 5.")

    # --------------------
    # p-values empíricos
    # --------------------
    out_p = {}
    for col in ["zmax_abs"] + [f"count_abs_ge_{th:g}" for th in Z_THRESHOLDS]:
        cvals = controls[col].to_numpy(dtype=float)
        out_p[f"p_empirical_{col}"] = empirical_pvalue(cvals, float(ev_stats[col]))

    # --------------------
    # OUTPUT (como pediste)
    # --------------------
    out_dir = resultados / f"{event}_nulltest"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "event_vs_controls_summary.csv"
    df.to_csv(summary_csv, index=False)

    pvals_txt = out_dir / "event_vs_controls_pvalues.txt"
    with open(pvals_txt, "w", encoding="utf-8") as f:
        f.write(f"EVENT VS CONTROLS — {event}\n")
        f.write(f"Controles válidos: {len(controls)}\n\n")
        for k, v in out_p.items():
            f.write(f"{k} = {v:.6g}\n")

    # --------------------
    # Figura: histogramas
    # --------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    plot_cols = ["zmax_abs"] + [f"count_abs_ge_{th:g}" for th in Z_THRESHOLDS]
    titles = ["max(|z|)", "N(|z|>=3)", "N(|z|>=4)", "N(|z|>=5)"]

    for ax, col, title in zip(axes, plot_cols, titles):
        cvals = controls[col].to_numpy(dtype=float)
        ax.hist(cvals, bins=12, alpha=0.75)
        ax.axvline(ev_stats[col], linestyle="--", linewidth=2)
        ax.set_title(f"{title}\n(p={out_p['p_empirical_'+col]:.4f})")
        ax.set_xlabel(col)
        ax.set_ylabel("freq")

    fig.suptitle(f"Null empírico: EVENTO vs CONTROLES — {event}", y=1.02)
    fig.tight_layout()

    fig_png = out_dir / "event_vs_controls.png"
    fig.savefig(fig_png, dpi=160)
    plt.close(fig)

    # --------------------
    # Consola
    # --------------------
    print("\n==============================")
    print(f"EVENT VS CONTROLS — {event}")
    print("------------------------------")
    print(f"Evento zmax(|z|): {ev_stats['zmax_abs']:.3f}")
    for th in Z_THRESHOLDS:
        print(f"Evento N(|z|>={th:g}): {ev_stats[f'count_abs_ge_{th:g}']}")
    print("\nP-valores (empírico, controles):")
    for k, v in out_p.items():
        print(f"  {k}: {v:.6g}")
    print("------------------------------")
    print(f"[OK] CSV: {summary_csv}")
    print(f"[OK] TXT: {pvals_txt}")
    print(f"[OK] PNG: {fig_png}")
    print("==============================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Null test EVENTO vs CONTROLES (empírico): compara métricas del evento contra días control."
    )
    parser.add_argument(
        "--event",
        "-e",
        dest="event",
        help="Nombre del evento (p.ej. tohoku2011, maule2010).",
    )
    parser.add_argument(
        "event_pos",
        nargs="?",
        help="(Compat) Evento como argumento posicional si no usas --event.",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        default="resultados",
        help="Carpeta base de resultados (default: resultados).",
    )

    args = parser.parse_args()
    event = args.event or args.event_pos
    if not event:
        raise SystemExit("Uso: python 11_null_event_vs_controls.py --event <EVENT>  (o posicional <EVENT>)")

    # Permite cambiar la carpeta base sin tocar el código
    # (main() usa project_root()/resultados por defecto, así que lo sobreescribimos con cwd temporal)
    # Implementación: si results-dir no es 'resultados', hacemos chdir al root y seteamos env var.
    # Para mantener el script simple y compatible, pasamos results_dir via variable global.
    RESULTS_DIR_OVERRIDE = args.results_dir  # noqa: N806

    main(event, results_dir=RESULTS_DIR_OVERRIDE)
