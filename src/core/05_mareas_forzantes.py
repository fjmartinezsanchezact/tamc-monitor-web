# ============================================================
# 05_mareas_forzantes.py (FIXED + FALLBACK)
# Forzantes TAMC + marea sintética
# - CSV -> resultados/<EVENTO_FULL>/forzantes/
# - PNG -> resultados/<EVENTO_FULL>/plots/
#
# FIX:
# - EVENTO puede contener "/" (padre/mainshock, padre/control_XX...)
# - EVENT_TAG se usa para nombres de archivo (sin "/")
# - t=0 (mainshock) se intenta leer de src/events.csv usando EVENT_BASE
# - Si no está en events.csv, fallback: extrae YYYYMMDD_HHMMSS del nombre de carpeta
# ============================================================

import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BIN_HOURS = 0.5
SMOOTH_WIN = 3


def synthetic_tide(hours):
    """Marea sintética simple (componentes principales)."""
    t = hours * 3600.0
    components = {
        "M2": (12.42 * 3600, 1.0),
        "S2": (12.00 * 3600, 0.5),
        "K1": (23.93 * 3600, 0.3),
        "O1": (25.82 * 3600, 0.2),
    }
    tide = np.zeros_like(t, dtype=float)
    for period, amp in components.values():
        tide += amp * np.sin(2 * np.pi * t / period)
    return tide


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_events_csv(events_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_event_column(df: pd.DataFrame) -> str:
    for c in ["event", "evento", "name", "event_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"events.csv no tiene columna de evento. Columnas: {list(df.columns)}")


def find_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "origin_time", "origin_time_utc",
        "mainshock_time", "mainshock_time_utc",
        "time", "time_utc", "datetime", "date_time",
        "fecha", "hora", "fecha_hora"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in ["origin", "mainshock", "time", "date", "fecha", "hora"]):
            return c
    raise KeyError(f"events.csv no tiene columna de tiempo/origen. Columnas: {list(df.columns)}")


def ensure_utc(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def parse_mainshock_from_event_name(event_base: str) -> pd.Timestamp:
    """
    Fallback: extrae YYYYMMDD_HHMMSS desde el nombre del evento base.
    Ej: 2017_Tehuantepec_..._20170908_044919  -> 2017-09-08 04:49:19 UTC
    """
    m = re.search(r"(\d{8})_(\d{6})", event_base)
    if not m:
        raise ValueError(
            f"No pude extraer timestamp YYYYMMDD_HHMMSS desde '{event_base}'. "
            "Necesito events.csv correcto o un nombre de carpeta que contenga fecha."
        )
    ymd, hms = m.group(1), m.group(2)
    s = f"{ymd}{hms}"  # YYYYMMDDHHMMSS
    # utc=True para que sea tz-aware
    t = pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"No pude parsear timestamp '{s}' del nombre '{event_base}'")
    return t


def load_mainshock_time(events_csv: Path, event_base: str) -> pd.Timestamp:
    """
    1) Intenta leer mainshock desde events.csv (match exacto por event_base).
    2) Si no lo encuentra, fallback: parsea desde el nombre del directorio (YYYYMMDD_HHMMSS).
    """
    if events_csv.exists():
        df = read_events_csv(events_csv)
        evcol = find_event_column(df)
        tcol = find_time_column(df)

        row = df[df[evcol].astype(str) == str(event_base)]
        if not row.empty:
            t = pd.to_datetime(row.iloc[0][tcol], utc=True, errors="coerce")
            if not pd.isna(t):
                return t

    # fallback
    return parse_mainshock_from_event_name(event_base)


def main():
    if len(sys.argv) < 2:
        print("Uso: python 05_mareas_forzantes.py EVENTO")
        sys.exit(1)

    EVENTO_FULL = sys.argv[1].strip()
    EVENT_TAG = EVENTO_FULL.replace("/", "_").replace("\\", "_")
    EVENTO_BASE = EVENTO_FULL.split("/")[0].split("\\")[0]

    print(f"\n[05] Forzantes + marea sintética — {EVENTO_FULL}")
    print(f"[05] Evento base (t=0): {EVENTO_BASE}")

    base = project_root()
    res_dir = base / "resultados" / EVENTO_FULL

    metrics_file = res_dir / "metrics" / "tamc_24h_metrics_allstations.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)

    events_csv = base / "src" / "events.csv"

    # ✅ t=0: events.csv si matchea; si no, lo saca del nombre del evento
    mainshock = load_mainshock_time(events_csv, EVENTO_BASE)
    print(f"[05] mainshock_utc = {mainshock}")

    df = pd.read_csv(metrics_file)

    if "time_center_iso" not in df.columns:
        raise KeyError("Falta 'time_center_iso' en metrics_allstations")
    if "zscore" not in df.columns and "z_score" in df.columns:
        df["zscore"] = df["z_score"]
    if "zscore" not in df.columns:
        raise KeyError("Falta 'zscore' (o 'z_score') en metrics_allstations")

    df["time"] = ensure_utc(df["time_center_iso"])
    df = df.dropna(subset=["time", "zscore"]).copy()

    # tiempo relativo
    df["t_rel_h"] = (df["time"] - mainshock).dt.total_seconds() / 3600.0
    df["z_abs"] = df["zscore"].astype(float).abs()

    # binning
    df["bin"] = (df["t_rel_h"] / BIN_HOURS).round().astype(int) * BIN_HOURS

    agg = (
        df.groupby("bin")["z_abs"]
        .agg(
            mean_z="mean",
            p90_z=lambda x: np.percentile(x, 90),
        )
        .reset_index()
        .sort_values("bin")
    )

    # suavizado
    agg["mean_z_smooth"] = (
        agg["mean_z"]
        .rolling(SMOOTH_WIN, center=True, min_periods=1)
        .median()
    )

    # marea sintética
    hours = agg["bin"].values
    tide = synthetic_tide(hours)
    tide_norm = (tide - tide.mean()) / tide.std()
    agg["tide_norm"] = tide_norm

    # ✅ CSV en /forzantes, PNG en /plots
    out_csv_dir = res_dir / "forzantes"
    out_png_dir = res_dir / "plots"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_csv_dir / f"forzantes_{EVENT_TAG}_24h.csv"
    out_png = out_png_dir / f"forzantes_{EVENT_TAG}_24h.png"

    agg.to_csv(out_csv, index=False)

    # plot
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.scatter(agg["bin"], agg["p90_z"], s=18, alpha=0.6, label="p90(|z|) TAMC")
    ax1.plot(agg["bin"], agg["mean_z"], lw=1.5, label="mean(|z|) TAMC")
    ax1.plot(agg["bin"], agg["mean_z_smooth"], lw=2.0, label="mean(|z|) suavizado")

    ax1.set_xlabel("Tiempo relativo (h) (0 = mainshock)")
    ax1.set_ylabel("|z| TAMC")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agg["bin"], agg["tide_norm"], lw=2, alpha=0.7, label="Marea sintética (norm.)")
    ax2.set_ylabel("Marea sintética (normalizada)")

    ax1.axvline(0, ls="--", color="k", alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"Forzantes TAMC + marea sintética — {EVENTO_FULL}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] Guardado:\n     {out_png}\n     {out_csv}")


if __name__ == "__main__":
    main()


