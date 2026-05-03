#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

import re
from obspy.geodetics.base import gps2dist_azimuth

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_METRICS_NAME = "tamc_24h_metrics_allstations.csv"
Z_THRESHOLDS = [3.0, 4.0, 5.0]


# -------------------------
# LOSO helpers (azimuth sector)
# -------------------------
def _pick_station_latlon_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Try to infer station latitude/longitude column names in the metrics CSV."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    lat_candidates = [
        "stalat", "sta_lat", "station_lat", "stationlatitude", "station_latitude",
        "latitude_station", "lat_station", "lat_sta", "sta.latitude", "stalatitude",
        "stationlat",
    ]
    lon_candidates = [
        "stalon", "sta_lon", "station_lon", "stationlongitude", "station_longitude",
        "longitude_station", "lon_station", "lon_sta", "sta.longitude", "stalongitude",
        "stationlon",
    ]

    lat_col = None
    lon_col = None

    for k in lat_candidates:
        if k in low:
            lat_col = low[k]
            break
    for k in lon_candidates:
        if k in low:
            lon_col = low[k]
            break

    # Heuristic fallback: look for something containing both ("sta" or "station") and ("lat"/"lon")
    if lat_col is None:
        for c in cols:
            cl = c.lower()
            if ("lat" in cl) and (("sta" in cl) or ("station" in cl)):
                lat_col = c
                break
    if lon_col is None:
        for c in cols:
            cl = c.lower()
            if (("lon" in cl) or ("lng" in cl) or ("long" in cl)) and (("sta" in cl) or ("station" in cl)):
                lon_col = c
                break

    if lat_col is None or lon_col is None:
        raise KeyError(
            "Para usar LOSO por azimut necesito columnas de lat/lon de estación en el CSV. "
            f"Columnas disponibles: {cols}"
        )
    return lat_col, lon_col


def _load_event_latlon(project_root_dir: Path, event_name: str) -> tuple[float, float] | None:
    """Try to load event LAT/LON from runlog.txt in data/<event>/runlog.txt (or data/otros/<event>/runlog.txt)."""
    candidates = [
        project_root_dir / "data" / event_name / "runlog.txt",
        project_root_dir / "tamcsismico" / "data" / event_name / "runlog.txt",
        project_root_dir / "data" / "otros" / event_name / "runlog.txt",
        project_root_dir / "tamcsismico" / "data" / "otros" / event_name / "runlog.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            mlat = re.search(r"^LAT=([+-]?[0-9]*\.?[0-9]+)", txt, flags=re.MULTILINE)
            mlon = re.search(r"^LON=([+-]?[0-9]*\.?[0-9]+)", txt, flags=re.MULTILINE)
            if mlat and mlon:
                return float(mlat.group(1)), float(mlon.group(1))
        except Exception:
            pass
    return None


def _apply_exclude_azimuth(df: pd.DataFrame, ev_lat: float, ev_lon: float, az0: float, az1: float) -> pd.DataFrame:
    """Exclude rows (stations) whose azimuth from event is in [az0, az1)."""
    lat_col, lon_col = _pick_station_latlon_cols(df)
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)

    keep_mask = []
    for la, lo in zip(lat, lon):
        if not (np.isfinite(la) and np.isfinite(lo)):
            keep_mask.append(False)
            continue
        _, az, _ = gps2dist_azimuth(ev_lat, ev_lon, float(la), float(lo))
        keep_mask.append(not (az0 <= az < az1))

    return df.loc[keep_mask].copy()


def project_root() -> Path:
    # .../tamcsismico/src/core/*.py -> parents[2] = .../tamcsismico
    return Path(__file__).resolve().parents[2]


def list_control_dirs_inside_event(event_dir: Path) -> list[Path]:
    """
    En tu layout: resultados/<EVENTO>/control_*/
    """
    out: list[Path] = []
    for d in event_dir.iterdir():
        if not d.is_dir():
            continue
        nm = d.name.lower()
        if "runlogs" in nm:
            continue
        if nm.startswith("control_"):
            out.append(d)
    out.sort(key=lambda p: p.name)
    return out


def find_metrics_csv_anywhere(folder: Path) -> Path:
    """
    Para controles: busca el CSV en folder/metrics/<file> o en cualquier subcarpeta.
    """
    p1 = folder / "metrics" / TARGET_METRICS_NAME
    if p1.exists():
        return p1

    matches = []
    for p in folder.rglob("*.csv"):
        if p.is_file() and p.name.lower() == TARGET_METRICS_NAME.lower():
            matches.append(p)

    if not matches:
        raise FileNotFoundError(f"No encontré '{TARGET_METRICS_NAME}' dentro de: {folder}")

    matches.sort(key=lambda x: len(x.parts))
    return matches[0]


def find_event_metrics_excluding_controls(event_dir: Path) -> Path:
    """
    Busca métricas del EVENTO dentro de event_dir PERO EXCLUYENDO subcarpetas control_*.

    Regla:
      1) event_dir/metrics/<file>
      2) búsqueda recursiva, ignorando rutas con un segmento control_*
    """
    p1 = event_dir / "metrics" / TARGET_METRICS_NAME
    if p1.exists():
        return p1

    matches: list[Path] = []
    for p in event_dir.rglob("*.csv"):
        if not (p.is_file() and p.name.lower() == TARGET_METRICS_NAME.lower()):
            continue

        # Si cualquier parte de la ruta es control_*, NO es el evento
        parts_lower = [seg.lower() for seg in p.parts]
        if any(seg.startswith("control_") for seg in parts_lower):
            continue

        matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"No encontré métricas del EVENTO '{TARGET_METRICS_NAME}' dentro de:\n{event_dir}\n\n"
            f"Necesitas generar el CSV del evento (no solo de los controles), por ejemplo:\n"
            f"  {event_dir}\\metrics\\{TARGET_METRICS_NAME}\n\n"
            f"O pasa la ruta exacta con:\n"
            f"  --event-metrics RUTA\\A\\{TARGET_METRICS_NAME}\n"
        )

    matches.sort(key=lambda x: len(x.parts))
    return matches[0]


def pick_z_column(df: pd.DataFrame) -> str:
    if "zscore" in df.columns:
        return "zscore"
    if "z_score" in df.columns:
        return "z_score"
    for c in df.columns:
        if "z" in c.lower():
            return c
    raise KeyError(f"No encuentro columna zscore/z_score. Columnas: {list(df.columns)}")


def compute_stats_from_metrics(csv_path: Path, *, exclude_azimuth: tuple[float, float] | None = None, ev_lat: float | None = None, ev_lon: float | None = None) -> dict:
    df = pd.read_csv(csv_path)

    # LOSO / excluir sector por azimut (opcional)
    if exclude_azimuth is not None:
        if ev_lat is None or ev_lon is None:
            raise ValueError("exclude_azimuth requiere ev_lat y ev_lon")
        az0, az1 = exclude_azimuth
        before_n = len(df)
        df = _apply_exclude_azimuth(df, float(ev_lat), float(ev_lon), float(az0), float(az1))
        after_n = len(df)
        if after_n == 0:
            raise ValueError(f"Tras excluir azimut [{az0},{az1}) no quedan filas/estaciones en {csv_path}")
        # Nota: no imprimimos aquí para no ensuciar salida; lo reporta main.

    zcol = pick_z_column(df)
    z = pd.to_numeric(df[zcol], errors="coerce").to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError(f"{csv_path}: serie z vacía o no numérica")

    zabs = np.abs(z)
    out = {
        "zmax_abs": float(np.max(zabs)),
        "n_points": int(zabs.size),
        "zcol": zcol,
    }
    for th in Z_THRESHOLDS:
        out[f"count_abs_ge_{th:g}"] = int(np.sum(zabs >= th))
    return out


def empirical_pvalue(controls: np.ndarray, event_value: float) -> float:
    """
    p empírico conservador (+1 smoothing):
      p = ( #{controls >= event} + 1 ) / (N + 1)
    """
    controls = np.asarray(controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    n = int(controls.size)
    if n == 0 or not np.isfinite(event_value):
        return float("nan")
    return float((np.sum(controls >= event_value) + 1) / (n + 1))


def main(event: str, results_dir: str = "resultados", event_metrics: str | None = None, show_png: bool = True,
         exclude_azimuth: tuple[float, float] | None = None, ev_lat: float | None = None, ev_lon: float | None = None) -> None:
    root = project_root()
    resultados = root / results_dir
    event_dir = resultados / event

    if not event_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de evento: {event_dir}")


    # --- LOSO: resolver lat/lon del evento si hace falta ---
    if exclude_azimuth is not None:
        if ev_lat is None or ev_lon is None:
            ll = _load_event_latlon(root, event)
            if ll is not None:
                ev_lat, ev_lon = ll
        if ev_lat is None or ev_lon is None:
            raise ValueError(
                "Para LOSO (--exclude-azimuth/--loso-oeste) necesito lat/lon del evento. "
                "Pásalas con --ev-lat/--ev-lon o asegúrate de tener data/<EVENTO>/runlog.txt con LAT/LON."
            )

    # --- EVENT METRICS ---
    if event_metrics:
        event_csv = Path(event_metrics)
        if not event_csv.exists():
            raise FileNotFoundError(f"--event-metrics no existe: {event_csv}")
    else:
        event_csv = find_event_metrics_excluding_controls(event_dir)

    ev_stats = compute_stats_from_metrics(event_csv, exclude_azimuth=exclude_azimuth, ev_lat=ev_lat, ev_lon=ev_lon)

    # --- CONTROLS ---
    control_dirs = list_control_dirs_inside_event(event_dir)
    if not control_dirs:
        raise RuntimeError(f"No encontré carpetas control_* dentro de: {event_dir}")

    print(f"[OK] EVENT_DIR:     {event_dir}")
    print(f"[OK] EVENT_METRICS: {event_csv}")
    print(f"[OK] Controles encontrados: {len(control_dirs)}")
    if exclude_azimuth is not None:
        az0, az1 = exclude_azimuth
        print(f"[LOSO] Excluyendo estaciones con azimut en [{az0:g}, {az1:g}) grados (respecto al epicentro).")

    rows = [{
        "group": "EVENT",
        "name": event,
        "metrics_csv": str(event_csv),
        **ev_stats
    }]

    skipped = 0
    for cd in control_dirs:
        try:
            c_csv = find_metrics_csv_anywhere(cd)
            c_stats = compute_stats_from_metrics(c_csv, exclude_azimuth=exclude_azimuth, ev_lat=ev_lat, ev_lon=ev_lon)
            rows.append({
                "group": "CONTROL",
                "name": cd.name,
                "metrics_csv": str(c_csv),
                **c_stats
            })
        except Exception as e:
            skipped += 1
            print(f"[WARN] Saltando {cd.name}: {e}")

    df = pd.DataFrame(rows)
    controls = df[df["group"] == "CONTROL"].copy()

    if len(controls) < 3:
        print(f"[WARN] Pocos controles válidos ({len(controls)}). Ideal >= 5.")
    if skipped:
        print(f"[WARN] Controles saltados: {skipped}")

    # --- P-VALUES ---
    metrics = ["zmax_abs"] + [f"count_abs_ge_{th:g}" for th in Z_THRESHOLDS]
    out_p = {}
    for col in metrics:
        cvals = pd.to_numeric(controls[col], errors="coerce").to_numpy(dtype=float)
        out_p[f"p_empirical_{col}"] = empirical_pvalue(cvals, float(ev_stats[col]))

    # --- OUTPUT INSIDE EVENT ---
    # Por defecto: resultados/<EVENTO>/nulltest
    # Si LOSO está activo: resultados/<EVENTO>/nulltest/LOSO_<AZ0>_<AZ1>
    out_dir = event_dir / "nulltest"
    if exclude_azimuth is not None:
        az0, az1 = exclude_azimuth
        out_dir = out_dir / f"LOSO_{az0:g}_{az1:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "event_vs_controls_summary.csv"
    df.to_csv(summary_csv, index=False)

    pvals_txt = out_dir / "event_vs_controls_pvalues.txt"
    with open(pvals_txt, "w", encoding="utf-8") as f:
        f.write(f"EVENT VS CONTROLS — {event}\n")
        f.write(f"EVENT_METRICS: {event_csv}\n")
        f.write(f"zcol_event: {ev_stats.get('zcol','')}\n")
        f.write(f"controles_found: {len(control_dirs)}\n")
        f.write(f"controles_validos: {len(controls)}\n")
        f.write(f"controles_saltados: {skipped}\n\n")
        for k, v in out_p.items():
            f.write(f"{k} = {v:.6g}\n")

    # --- FIGURA ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    titles = ["max(|z|)", "N(|z|>=3)", "N(|z|>=4)", "N(|z|>=5)"]

    for ax, col, title in zip(axes, metrics, titles):
        vals = pd.to_numeric(controls[col], errors="coerce").dropna().to_numpy(dtype=float)
        ax.hist(vals, bins=15, alpha=0.75)
        ax.axvline(ev_stats[col], linestyle="--", linewidth=2)
        ax.set_title(f"{title}\n(p={out_p['p_empirical_'+col]:.4f})")
        ax.set_xlabel(col)
        ax.set_ylabel("freq")

    fig.suptitle(f"Null empírico: EVENTO vs CONTROLES — {event}", y=1.02)
    fig.tight_layout()

    fig_png = out_dir / "event_vs_controls.png"
    fig.savefig(fig_png, dpi=160)
    plt.close(fig)

    # --- CONSOLA ---
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
    print(f"[OK] OUT_DIR: {out_dir}")
    print(f"[OK] CSV:     {summary_csv}")
    print(f"[OK] TXT:     {pvals_txt}")
    print(f"[OK] PNG:     {fig_png}")
    print("==============================\n")

    # --- MOSTRAR PNG POR PANTALLA (Windows) ---
    if show_png and sys.platform.startswith("win"):
        try:
            os.startfile(fig_png)  # abre con el visor por defecto
        except Exception as e:
            print(f"[WARN] No pude abrir el PNG automáticamente: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="11 — Event vs Controls (controles dentro del evento)")
    ap.add_argument("--event", "-e", dest="event_flag", default=None)
    ap.add_argument("event_pos", nargs="?", default=None)
    ap.add_argument("--results-dir", "-r", default="resultados")
    ap.add_argument("--event-metrics", default=None, help="Ruta exacta al CSV del EVENTO (allstations)")
    ap.add_argument("--no-show", action="store_true", help="No abrir el PNG al finalizar (por defecto se abre)")

    ap.add_argument(
        "--exclude-azimuth",
        nargs=2,
        type=float,
        metavar=("AZ_MIN", "AZ_MAX"),
        default=None,
        help="LOSO: excluir estaciones cuyo azimut (desde el evento) esté en [AZ_MIN, AZ_MAX) grados."
    )
    ap.add_argument(
        "--loso-oeste",
        action="store_true",
        help="Atajo LOSO: excluye sector Oeste (Sol-Oeste) azimut [225, 315)."
    )
    ap.add_argument(
        "--ev-lat",
        type=float,
        default=None,
        help="Latitud del evento (solo necesario si usas --exclude-azimuth/--loso-oeste y no hay runlog en data/)."
    )
    ap.add_argument(
        "--ev-lon",
        type=float,
        default=None,
        help="Longitud del evento (solo necesario si usas --exclude-azimuth/--loso-oeste y no hay runlog en data/)."
    )

    args = ap.parse_args()

    ev = args.event_flag or args.event_pos
    if not ev:
        raise SystemExit("Uso: python 11_null_event_vs_controls.py --event <EVENTO>  (o posicional <EVENTO>)")

    main(
        ev,
        results_dir=args.results_dir,
        event_metrics=args.event_metrics,
        show_png=(not args.no_show),
        exclude_azimuth=(tuple(args.exclude_azimuth) if args.exclude_azimuth is not None else ( (225.0, 315.0) if args.loso_oeste else None )),
        ev_lat=args.ev_lat,
        ev_lon=args.ev_lon,
    )


# OUTDIR: LOSO results go under resultados/<EVENTO>/nulltest/LOSO_*
