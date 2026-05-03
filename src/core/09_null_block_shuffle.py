#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09 — Null test por block shuffle temporal.
Rompe la alineación temporal preservando autocorrelación (shuffle por bloques)
y compara un estadístico de estructura temporal observado con una distribución
nula por shuffles.
CORRECCIONES IMPORTANTES:
- Detecta si el CSV es un "resumen" (metric,real_value) y NO sirve.
- Busca automáticamente un CSV de series (timeseries) dentro de resultados/<event>/...
  priorizando mainshock/ y evitando control_ cuando sea posible.
- Acepta alias de columna para zscore (z, absz, z_abs, etc.)
- Cambia el estadístico por defecto a uno dependiente del orden temporal:
  longest_run(|z| >= zthr). Esto hace que el block-shuffle tenga sentido.
- Permite escoger estadístico vía CLI: --stat {longest_run, mean_abs, max_abs}
  (max_abs se deja por compatibilidad, pero NO es recomendable para shuffle).
Salida:
- out_dir/block_shuffle_summary.csv con stat, stat_event, p_empirical, etc.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from obspy import UTCDateTime
# IMPORTA _common (si lo tienes) pero no dependemos ciegamente de su salida
from _common import default_metrics_file, default_test_out_dir, get_mainshock_time
# -------------------------
# Fallback robusto mainshock
# -------------------------
def infer_mainshock_time_from_event_name(event: str) -> UTCDateTime:
    """
    Extrae mainshock desde el nombre del evento:
    ..._YYYYMMDD_HHMMSS
    """
    m = re.search(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)", event)
    if not m:
        raise ValueError(f"No se pudo inferir mainshock desde el nombre del evento: {event}")
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return UTCDateTime(dt)
def safe_get_mainshock_time(event: str) -> UTCDateTime:
    """
    Wrapper definitivo: nunca deja propagar el error.
    """
    try:
        return get_mainshock_time(event)
    except Exception as e:
        ms = infer_mainshock_time_from_event_name(event)
        print(f"[WARN] get_mainshock_time falló ({e}). Uso fallback desde nombre del evento: {ms}.")
        return ms
# -------------------------
# Utilidades de búsqueda CSV
# -------------------------
def _project_root() -> Path:
    # src/core/09_null_block_shuffleA.py -> parents[0]=core, [1]=src, [2]=root
    return Path(__file__).resolve().parents[2]
def _looks_like_summary_metrics(df_head: pd.DataFrame) -> bool:
    cols = [c.strip().lower() for c in df_head.columns]
    return cols == ["metric", "real_value"]
def _pick_z_column(df: pd.DataFrame) -> str:
    """
    Elige una columna de z-score usando aliases comunes.
    Si no hay ninguna, lanza ValueError con lista de columnas.
    """
    candidates = [
        "zscore", "z_score", "z", "Z",
        "absz", "abs_z", "z_abs", "|z|",
        "zval", "z_value", "tamc_z", "z_tamc"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "No se encontró ninguna columna tipo z-score (zscore/z/absz/etc.). "
        f"Columnas disponibles: {list(df.columns)}"
    )
def _rank_timeseries_csv_path(p: Path) -> tuple:
    """
    Ranking para seleccionar CSV de series:
    1) prioriza mainshock/
    2) prioriza metrics/ en raíz del evento
    3) penaliza control_
    4) prioriza nombres conocidos (tamc_24h_metrics_allstations)
    """
    s = str(p).lower()
    is_mainshock = 0 if ("\\mainshock\\" in s or "/mainshock/" in s) else 1
    is_root_metrics = 0 if re.search(r"[\\/]\bmetrics[\\/]", s) and ("control_" not in s) else 1
    is_control = 1 if ("\\control_" in s or "/control_" in s or "\\control\\" in s or "/control/" in s) else 0
    is_preferred_name = 0 if "tamc_24h_metrics_allstations" in p.name.lower() else 1
    # length as tiebreaker (shorter path tends to be "root" not deep)
    return (is_mainshock, is_preferred_name, is_root_metrics, is_control, len(s))
def _find_timeseries_metrics_csv(event: str) -> Path | None:
    """
    Busca un CSV de series (no resumen) dentro de resultados/<event>/... que contenga
    una columna z-score plausible. Devuelve la mejor coincidencia.
    Heurística:
    - Recorre resultados/<event> recursivo
    - Para cada *.csv lee solo cabecera (nrows=5)
    - Descarta el resumen metric,real_value
    - Exige que haya columna z plausible
    - Ordena por ranking (_rank_timeseries_csv_path)
    """
    root = _project_root()
    event_dir = root / "resultados" / event
    if not event_dir.exists():
        return None
    csvs = list(event_dir.rglob("*.csv"))
    if not csvs:
        return None
    valid = []
    for p in csvs:
        try:
            head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if _looks_like_summary_metrics(head):
            continue
        try:
            _ = _pick_z_column(head)
        except Exception:
            continue
        valid.append(p)
    if not valid:
        return None
    valid = sorted(valid, key=_rank_timeseries_csv_path)
    return valid[0]
# -------------------------
# Block shuffle
# -------------------------
def block_shuffle(arr: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle temporal por bloques contiguos.
    """
    n = len(arr)
    if block_size <= 0:
        raise ValueError("block_size debe ser > 0")
    blocks = [arr[i:i + block_size] for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]
# -------------------------
# Estadísticos (dependientes del orden)
# -------------------------
def longest_run(binary: np.ndarray) -> int:
    """Longest consecutive run of ones in a 0/1 array."""
    best = 0
    cur = 0
    for v in binary:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best
def stat_longest_run_absz(z: np.ndarray, zthr: float) -> int:
    """Longest run where |z| >= zthr."""
    active = (np.abs(z) >= zthr)
    return int(longest_run(active.astype(np.int8)))
def stat_mean_absz(z: np.ndarray) -> float:
    """Mean of |z| (sí depende del conjunto, pero menos sensible que longest_run al orden)."""
    return float(np.nanmean(np.abs(z)))
def stat_max_absz(z: np.ndarray) -> float:
    """
    Max of |z|. AVISO: esto es (casi) invariante al shuffle y puede dar p~1.
    Se deja solo por compatibilidad/debug.
    """
    return float(np.nanmax(np.abs(z)))
def compute_stat(z: np.ndarray, stat_name: str, zthr: float) -> float:
    stat_name = stat_name.lower().strip()
    if stat_name == "longest_run":
        return float(stat_longest_run_absz(z, zthr=zthr))
    if stat_name == "mean_abs":
        return float(stat_mean_absz(z))
    if stat_name == "max_abs":
        return float(stat_max_absz(z))
    raise ValueError("stat desconocido. Usa: longest_run, mean_abs, max_abs")
# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--block-size", type=int, default=60)
    ap.add_argument("--n-shuffles", type=int, default=1000)
    ap.add_argument(
        "--metrics-csv",
        default=None,
        help="Ruta explícita al CSV de series (timeseries) con zscore/z/absz. "
             "Si no se indica, se usa default_metrics_file(event) y si es resumen, se auto-busca."
    )
    # NUEVO: estadístico temporal
    ap.add_argument(
        "--stat",
        default="longest_run",
        choices=["longest_run", "mean_abs", "max_abs"],
        help="Estadístico a testear bajo block-shuffle. Recomendado: longest_run."
    )
    ap.add_argument(
        "--zthr",
        type=float,
        default=5.0,
        help="Umbral para longest_run: cuenta rachas de |z|>=zthr (default=5.0)."
    )
    # reproducibilidad
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    event = args.event
    # === RESOLUCIÓN MAINSHOCK (ROBUSTA) ===
    mainshock = safe_get_mainshock_time(event)
    # === Resolver metrics CSV ===
    if args.metrics_csv:
        metrics_csv = Path(args.metrics_csv).expanduser().resolve()
        if not metrics_csv.exists():
            raise FileNotFoundError(f"No existe metrics CSV (forzado): {metrics_csv}")
    else:
        metrics_csv = default_metrics_file(event)
        if not Path(metrics_csv).exists():
            print(f"[WARN] No existe metrics CSV (default): {metrics_csv}")
            alt = _find_timeseries_metrics_csv(event)
            if alt is None:
                raise FileNotFoundError(
                    "No existe metrics CSV (default) y no se encontró automáticamente ningún CSV de series "
                    "(timeseries) bajo resultados/<event>/.\n"
                    "Solución: (1) genera/copias un alias real_metrics.csv en la raíz del evento, o "
                    "(2) indica explícitamente el CSV con --metrics-csv <ruta>."
                )
            print(f"[OK] Usaré CSV de series encontrado automáticamente: {alt}")
            metrics_csv = alt
    # Leer cabecera para ver si es resumen
    try:
        head = pd.read_csv(metrics_csv, nrows=5)
    except Exception as e:
        raise ValueError(f"No se pudo leer metrics CSV: {metrics_csv} ({e})")
    if _looks_like_summary_metrics(head):
        print(f"[WARN] El metrics CSV detectado es un RESUMEN (metric,real_value): {metrics_csv}")
        alt = _find_timeseries_metrics_csv(event)
        if alt is None:
            raise ValueError(
                "El archivo de métricas detectado es un resumen (metric,real_value) y NO sirve para block-shuffle.\n"
                "No se encontró automáticamente ningún CSV de series (timeseries) bajo resultados/<event>/.\n"
                "Solución: indica explícitamente el CSV de series con --metrics-csv <ruta>."
            )
        print(f"[OK] Usaré CSV de series encontrado automáticamente: {alt}")
        metrics_csv = alt
    # === Output dir ===
    out_dir = Path(args.out_dir) if args.out_dir else default_test_out_dir(event, "09_block_shuffle")
    out_dir.mkdir(parents=True, exist_ok=True)
    # === Cargar datos completos ===
    df = pd.read_csv(metrics_csv)
    # Elegir columna z
    zcol = _pick_z_column(df)
    # Normalizar a df['zscore'] para no tocar el resto del script
    if zcol != "zscore":
        df["zscore"] = pd.to_numeric(df[zcol], errors="coerce")
    else:
        df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")
    if df["zscore"].isna().all():
        raise ValueError(
            f"La columna seleccionada ('{zcol}') existe pero tras convertir a numérico todo es NaN.\n"
            f"Archivo: {metrics_csv}"
        )
    # Serie base
    z = df["zscore"].to_numpy(dtype=float)
    # === Estadístico del evento (depende del orden temporal) ===
    stat_name = args.stat
    zthr = float(args.zthr)
    stat_event = compute_stat(z, stat_name=stat_name, zthr=zthr)
    # === Distribución nula por shuffles ===
    rng = np.random.default_rng(args.seed)
    stat_null = np.empty(args.n_shuffles, dtype=float)
    for i in range(args.n_shuffles):
        z_shuf = block_shuffle(z, args.block_size, rng)
        stat_null[i] = compute_stat(z_shuf, stat_name=stat_name, zthr=zthr)
    # p-valor (cola superior: “más estructura” = valor más grande)
    # Nota: usamos >= por definición conservadora. Si tu stat es discreto (longest_run),
    # es normal que haya empates: eso hace el test más conservador.
    p_empirical = (np.sum(stat_null >= stat_event) + 1) / (len(stat_null) + 1)
    # Guardar resultados (incluye percentiles para inspección rápida)
    summary = pd.DataFrame(
        {
            "event": [event],
            "metrics_csv_used": [str(metrics_csv)],
            "z_column_used": [zcol],
            "stat": [stat_name],
            "zthr": [zthr],
            "stat_event": [stat_event],
            "p_empirical": [p_empirical],
            "null_q05": [float(np.quantile(stat_null, 0.05))],
            "null_q50": [float(np.quantile(stat_null, 0.50))],
            "null_q95": [float(np.quantile(stat_null, 0.95))],
            "n_shuffles": [args.n_shuffles],
            "block_size": [args.block_size],
            "seed": [args.seed],
            "mainshock": [str(mainshock)],
        }
    )
    summary.to_csv(out_dir / "block_shuffle_summary.csv", index=False)
    # (Opcional) guardar la distribución nula completa para figuras
    pd.DataFrame({"stat_null": stat_null}).to_csv(out_dir / "block_shuffle_null_distribution.csv", index=False)
    print("====================================")
    print(f"09 BLOCK SHUFFLE — {event}")
    print("------------------------------------")
    print(f"metrics_csv_used: {metrics_csv}")
    print(f"z_column_used: {zcol}")
    print(f"stat: {stat_name} (zthr={zthr} si aplica)")
    print(f"stat_event: {stat_event}")
    print(f"p_empirical: {p_empirical:.6f}")
    print(f"[OK] OUT_DIR: {out_dir}")
    print("====================================")
if __name__ == "__main__":
    main()
