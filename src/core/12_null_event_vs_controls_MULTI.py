
# 12_null_event_vs_controls_MULTI.py
# Ejecuta null-test "evento vs controles" para UNO o VARIOS eventos.
# Uso:
#   (tamc) ...\src\core> python 12_null_event_vs_controls_MULTI.py
#   (tamc) ...\src\core> python 12_null_event_vs_controls_MULTI.py tohoku2011
#   (tamc) ...\src\core> python 12_null_event_vs_controls_MULTI.py maule2010 tohoku2011

from __future__ import annotations


import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
METRICS_FILENAME = "tamc_24h_metrics_allstations.csv"

# Columnas posibles para el z-score (tú ya tienes 'zscore' en Tohoku).
Z_COL_CANDIDATES = [
    "zscore",
    "z_score",
    "z",
    "zrot",
    "z_rot",
    "z_score_rot",
    "zscore_rot",
]

# Estadísticos a comparar
THRESHOLDS = [3, 4, 5]


# -----------------------------
# Utils
# -----------------------------
def repo_root_from_this_file() -> Path:
    """
    Estamos en .../tamcsismico/src/core.
    Repo root = .../tamcsismico
    """
    here = Path(__file__).resolve()
    return here.parents[2]


def find_z_column(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in Z_COL_CANDIDATES:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fallback: cualquier columna que contenga "z"
    for c in df.columns:
        if "z" in c.lower():
            return c
    raise ValueError(
        f"No encontré columna zscore en el CSV. Columnas disponibles: {list(df.columns)}"
    )


def load_metrics(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path)
    zcol = find_z_column(df)
    df = df.copy()
    df[zcol] = pd.to_numeric(df[zcol], errors="coerce")
    df = df.dropna(subset=[zcol])
    return df, zcol


def compute_stats(df: pd.DataFrame, zcol: str) -> Dict[str, float]:
    z = df[zcol].to_numpy(dtype=float)
    zabs = np.abs(z)
    out = {
        "zmax_abs": float(np.nanmax(zabs)) if len(zabs) else float("nan"),
    }
    for thr in THRESHOLDS:
        out[f"count_abs_ge_{thr}"] = float(np.sum(zabs >= thr))
    return out


def empirical_pvalue(event_value: float, control_values: List[float]) -> float:
    """
    P-valor empírico (con corrección +1):
      p = (1 + #{control >= event}) / (N + 1)
    """
    vals = [v for v in control_values if v is not None and np.isfinite(v)]
    n = len(vals)
    if n == 0 or not np.isfinite(event_value):
        return float("nan")
    ge = sum(1 for v in vals if v >= event_value)
    return (1.0 + ge) / (n + 1.0)


def list_control_dirs(resultados_dir: Path, event_name: str) -> List[Path]:
    """
    Controles válidos si el nombre empieza por:
      - control_<event>_
      - <event>_control_day_
    """
    prefixes = [f"control_{event_name}_", f"{event_name}_control_day_"]
    dirs = []
    for p in resultados_dir.iterdir():
        if p.is_dir():
            name = p.name.lower()
            if any(name.startswith(pref.lower()) for pref in prefixes):
                dirs.append(p)
    dirs.sort(key=lambda x: x.name)
    return dirs


def plot_event_vs_controls(
    event_name: str,
    event_stats: Dict[str, float],
    controls_stats: Dict[str, List[float]],
    out_png: Path,
):
    metrics = ["zmax_abs"] + [f"count_abs_ge_{t}" for t in THRESHOLDS]
    titles = [
        "zmax(|z|)",
        "N(|z|>=3)",
        "N(|z|>=4)",
        "N(|z|>=5)",
    ]

    plt.figure(figsize=(12, 8))
    for i, (m, title) in enumerate(zip(metrics, titles), start=1):
        ax = plt.subplot(2, 2, i)
        vals = [v for v in controls_stats.get(m, []) if np.isfinite(v)]
        ev = event_stats.get(m, float("nan"))

        if len(vals) > 0:
            ax.hist(vals, bins=12)
        ax.axvline(ev, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(m)
        ax.set_ylabel("Controles (frecuencia)")

    plt.suptitle(f"EVENT VS CONTROLS — {event_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def read_controls_meta(root: Path, event_name: str) -> Dict[str, object]:
    """
    Lee tamcsismico/data/control_<event>_RUNLOGS/controls_meta.json si existe.
    Devuelve dict (vacío si no existe).
    """
    meta_path = root / "data" / f"control_{event_name}_RUNLOGS" / "controls_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Normalizamos algunos campos típicos (por si faltan)
        meta_out = {}
        for k in [
            "requested_controls",
            "used_controls",
            "probe_minutes",
            "t0_utc",
            "min_stations",
            "channels",
            "stations",
            "generated_at_utc",
            "event",
        ]:
            if k in meta:
                meta_out[k] = meta[k]
        meta_out["_controls_meta_path"] = str(meta_path)
        return meta_out
    except Exception:
        return {"_controls_meta_path": str(meta_path), "_controls_meta_error": True}


# -----------------------------
# Core per-event runner
# -----------------------------
def run_one_event(event_name: str) -> Dict[str, object]:
    root = repo_root_from_this_file()
    resultados_dir = root / "resultados"

    event_dir = resultados_dir / event_name
    event_metrics = event_dir / "metrics" / METRICS_FILENAME

    print(f"[OK] Evento: {event_name}")

    # Carga evento
    df_event, zcol_event = load_metrics(event_metrics)
    event_stats = compute_stats(df_event, zcol_event)

    # Controles
    control_dirs = list_control_dirs(resultados_dir, event_name)
    print(f"[OK] Controles encontrados: {len(control_dirs)}")

    controls_stats: Dict[str, List[float]] = {k: [] for k in event_stats.keys()}
    controls_used = 0

    used_controls_names: List[str] = []
    skipped_controls: List[Dict[str, str]] = []

    for cdir in control_dirs:
        cmetrics = cdir / "metrics" / METRICS_FILENAME
        if not cmetrics.exists():
            msg = f"No existe: {cmetrics}"
            print(f"[WARN] Saltando control {cdir.name}: {msg}")
            skipped_controls.append({"control": cdir.name, "reason": msg})
            continue
        try:
            df_c, zcol_c = load_metrics(cmetrics)
            cstats = compute_stats(df_c, zcol_c)
            for k in controls_stats.keys():
                controls_stats[k].append(float(cstats.get(k, float("nan"))))
            controls_used += 1
            used_controls_names.append(cdir.name)
        except Exception as e:
            print(f"[WARN] Saltando control {cdir.name}: {e}")
            skipped_controls.append({"control": cdir.name, "reason": str(e)})

    # P-valores empíricos
    pvals = {
        "p_empirical_zmax_abs": empirical_pvalue(
            event_stats["zmax_abs"], controls_stats["zmax_abs"]
        )
    }
    for thr in THRESHOLDS:
        key = f"count_abs_ge_{thr}"
        pvals[f"p_empirical_{key}"] = empirical_pvalue(event_stats[key], controls_stats[key])

    # Output folder
    out_dir = resultados_dir / f"{event_name}_nulltest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar lista usada / descartada (muy útil para auditoría)
    used_list_path = out_dir / "controls_used_list.txt"
    with open(used_list_path, "w", encoding="utf-8") as f:
        for name in used_controls_names:
            f.write(name + "\n")

    skipped_list_path = out_dir / "controls_skipped_list.txt"
    with open(skipped_list_path, "w", encoding="utf-8") as f:
        for item in skipped_controls:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Leer meta de descarga (si existe)
    controls_meta = read_controls_meta(root, event_name)

    # CSV resumen (1 fila)
    row = {
        "event": event_name,
        "zcol": zcol_event,
        "n_controls_found": len(control_dirs),
        "n_controls_used": controls_used,
        "controls_used_list": str(used_list_path),
        "controls_skipped_list": str(skipped_list_path),
        **{f"event_{k}": v for k, v in event_stats.items()},
        **pvals,
        # Metadatos de descarga (si están)
        **{f"download_{k}": v for k, v in controls_meta.items()},
    }

    summary_csv = out_dir / "event_vs_controls_summary.csv"
    pd.DataFrame([row]).to_csv(summary_csv, index=False)

    # TXT pvals
    ptxt = out_dir / "event_vs_controls_pvalues.txt"
    with open(ptxt, "w", encoding="utf-8") as f:
        f.write("EVENT VS CONTROLS\n")
        f.write(f"event: {event_name}\n")
        f.write(f"zcol: {zcol_event}\n")
        f.write(f"controls_found: {len(control_dirs)}\n")
        f.write(f"controls_used: {controls_used}\n")
        f.write(f"controls_used_list: {used_list_path}\n")
        f.write(f"controls_skipped_list: {skipped_list_path}\n")

        if controls_meta:
            f.write("\nDOWNLOAD META (from controls_meta.json)\n")
            for k, v in controls_meta.items():
                f.write(f"{k}: {v}\n")

        f.write("\nEVENT STATS\n")
        for k, v in event_stats.items():
            f.write(f"event_{k}: {v}\n")

        f.write("\nP-VALORES (empírico, controles)\n")
        for k, v in pvals.items():
            f.write(f"{k}: {v}\n")

    # PNG
    out_png = out_dir / "event_vs_controls.png"
    plot_event_vs_controls(event_name, event_stats, controls_stats, out_png)

    # Print resumen bonito
    print("\n==============================")
    print(f"EVENT VS CONTROLS — {event_name}")
    print("------------------------------")
    print(f"Evento zcol: {zcol_event}")
    print(f"Evento zmax(|z|): {event_stats['zmax_abs']}")
    for thr in THRESHOLDS:
        print(f"Evento N(|z|>={thr}): {event_stats[f'count_abs_ge_{thr}']}")
    print("\nP-valores (empírico, controles):")
    for k, v in pvals.items():
        print(f"  {k}: {v}")
    if controls_meta:
        req = controls_meta.get("requested_controls", None)
        used = controls_meta.get("used_controls", None)
        if req is not None or used is not None:
            print(f"\nDownload meta: requested_controls={req} used_controls={used}")
    print("------------------------------")
    print(f"[OK] CSV: {summary_csv}")
    print(f"[OK] TXT: {ptxt}")
    print(f"[OK] PNG: {out_png}")
    print(f"[OK] used list: {used_list_path}")
    print(f"[OK] skipped list: {skipped_list_path}")
    print("==============================\n")

    return row


def main(argv: List[str]) -> int:
    events = argv[1:] if len(argv) > 1 else ["maule2010", "tohoku2011"]

    rows = []
    for ev in events:
        try:
            rows.append(run_one_event(ev))
        except Exception as e:
            print(f"[ERROR] Falló evento {ev}: {e}")

    # Resumen multi-evento
    if len(rows) > 0:
        root = repo_root_from_this_file()
        out_dir = root / "resultados" / "multi_nulltest"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "multi_event_summary.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] Multi resumen CSV: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
