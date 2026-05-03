# 08_null_local_significance.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import get_mainshock_time, default_metrics_file, default_test_out_dir


def pick_z_column(df: pd.DataFrame) -> str:
    candidates = ["z_score", "zscore", "z", "z_rot", "z_tamc", "z_tamc_rot", "z_abs", "zmax_abs"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: busca columnas que contengan 'z'
    for c in df.columns:
        if "z" in c.lower():
            return c
    raise ValueError("No se encontró columna de z-score en el CSV.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="tohoku2011 | maule2010 | ...")
    ap.add_argument("--metrics-file", default=None, help="Override path al CSV de métricas")
    ap.add_argument("--out-dir", default=None, help="Override de directorio de salida")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-iter", type=int, default=20_000)
    args = ap.parse_args()

    event = args.event
    metrics_file = default_metrics_file(event) if args.metrics_file is None else Path(args.metrics_file)
    out_dir = default_test_out_dir(event, "08_null_local_significance") if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        raise FileNotFoundError(f"No se encuentra METRICS_FILE: {metrics_file}")

    df = pd.read_csv(metrics_file)
    zcol = pick_z_column(df)

    # Asumimos que el CSV tiene time_center_iso o algo equivalente
    time_col = "time_center_iso" if "time_center_iso" in df.columns else None
    if time_col is None:
        raise ValueError("Falta columna time_center_iso en el CSV de métricas.")

    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"])

    mainshock = get_mainshock_time(event)

    # Ventana precursora 24h (puedes hacerla argumento si quieres)
    window_hours = 24
    t0 = mainshock - pd.Timedelta(hours=window_hours)
    t1 = mainshock
    mask = (df["time"] >= t0) & (df["time"] < t1)

    z = np.abs(df.loc[mask, zcol].to_numpy(dtype=float))
    z_obs = float(np.max(z))

    rng = np.random.default_rng(args.seed)
    z_all = np.abs(df[zcol].to_numpy(dtype=float))

    # Null local: re-muestreo aleatorio del mismo tamaño
    n = len(z)
    null_max = []
    for _ in range(args.n_iter):
        sample = rng.choice(z_all, size=n, replace=False if n <= len(z_all) else True)
        null_max.append(np.max(sample))
    null_max = np.array(null_max, dtype=float)

    p = (np.sum(null_max >= z_obs) + 1) / (len(null_max) + 1)

    # Guardar
    np.save(out_dir / "null_local_max.npy", null_max)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"EVENT={event}\n")
        f.write(f"metrics_file={metrics_file}\n")
        f.write(f"zcol={zcol}\n")
        f.write(f"z_obs={z_obs}\n")
        f.write(f"n_iter={args.n_iter}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"p_value={p}\n")

    # Figura
    plt.figure()
    plt.hist(null_max, bins=50)
    plt.axvline(z_obs, linestyle="--")
    plt.title(f"08 Null local (max |z|) — {event} (p={p:.4g})")
    plt.xlabel("max |z| (null)")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.savefig(out_dir / "null_local_hist.png", dpi=150)
    plt.close()

    print(f"[OK] 08_null_local_significance — {event} p={p:.6f} OUT={out_dir}")


if __name__ == "__main__":
    from pathlib import Path
    main()

