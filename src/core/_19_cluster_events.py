
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_cluster_events.py
Earthquake clustering based on REAL inter-event summary features.

Reads:
  resultados/null_tests_13_14/analysis/real_multi_phase_summary.csv

Writes:
  resultados/earthquake_clustering/
    - features_used.csv
    - clustering_labels.csv
    - pca_projection.csv
    - pca_clusters.png
    - dendrogram.png (if scipy available)
    - summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_real_summary(root: Path) -> pd.DataFrame:
    """
    Load the REAL per-event feature table (wide format).
    Expected columns include at least: event + numeric metrics.
    """
    path = root / "resultados" / "null_tests_13_14" / "analysis" / "real_multi_phase_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro el CSV REAL para clustering:\n{path}\n"
            f"Primero corre inter-event (13..18) para generarlo."
        )
    df = pd.read_csv(path)
    if "event" not in df.columns:
        # fallback: sometimes could be named differently
        # try first column if it looks like event names
        raise ValueError(f"El CSV no contiene columna 'event'. Columnas: {list(df.columns)}")
    return df


def _filter_events(df: pd.DataFrame, selected_events: list[str] | None) -> pd.DataFrame:
    if not selected_events:
        return df.copy()
    sel = set(selected_events)
    out = df[df["event"].astype(str).isin(sel)].copy()
    return out


def _prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (features_df, feature_cols)
    features_df has columns: event + numeric feature columns
    """
    df = df.copy()

    # Keep only numeric columns (besides 'event')
    numeric_cols = []
    for c in df.columns:
        if c == "event":
            continue
        # try coercing to numeric
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() >= 2:  # at least 2 valid numbers
            df[c] = coerced
            numeric_cols.append(c)

    if len(numeric_cols) < 2:
        raise ValueError(
            "No hay suficientes columnas numéricas válidas para clusterizar.\n"
            f"Columnas numéricas detectadas: {numeric_cols}\n"
            f"Columnas totales: {list(df.columns)}"
        )

    # Drop rows with too many NaNs (require at least 2 numeric values)
    df["_n_valid"] = df[numeric_cols].notna().sum(axis=1)
    df = df[df["_n_valid"] >= 2].drop(columns=["_n_valid"])

    # Impute remaining NaNs with column medians
    for c in numeric_cols:
        med = float(df[c].median(skipna=True))
        df[c] = df[c].fillna(med)

    # Ensure we still have enough events
    if df.shape[0] < 2:
        raise ValueError("Necesito al menos 2 eventos con features válidas para clusterizar.")

    return df[["event"] + numeric_cols].copy(), numeric_cols


def _auto_k(n_events: int) -> int:
    # sensible default for small N
    if n_events <= 3:
        return 2
    if n_events <= 8:
        return 2
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Earthquake clustering (REAL inter-event features).")
    ap.add_argument("--root", type=str, default=".", help="Project root (tamcsismico).")
    ap.add_argument("--k", type=int, default=0, help="KMeans clusters (0=auto).")
    ap.add_argument("--no_plots", action="store_true", help="Do not save plots.")
    ap.add_argument("events", nargs="*", help="Optional list of event folder names to include.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / "resultados" / "earthquake_clustering"
    _safe_mkdir(outdir)

    # 1) Load wide REAL summary table
    try:
        df = _read_real_summary(root)
    except Exception as e:
        print(f"[ES] Error leyendo real_multi_phase_summary.csv: {e}")
        print("[EN] Failed reading real_multi_phase_summary.csv.")
        return 2

    # 2) Filter to selected events (if provided)
    df_sel = _filter_events(df, args.events)
    if df_sel.empty:
        print("[ES] La selección de eventos no coincide con el CSV REAL.")
        print("[EN] Selected events do not match REAL CSV.")
        print("Selected:", args.events)
        print("Available:", list(df["event"].astype(str).unique())[:50])
        return 3

    # 3) Build numeric feature matrix
    try:
        feats_df, feature_cols = _prepare_feature_matrix(df_sel)
    except Exception as e:
        print(f"[ES] No pude extraer features suficientes. Revisa el CSV REAL. Detalle: {e}")
        print("[EN] Could not extract enough numeric features. Check REAL CSV.")
        return 4

    # Save features used
    feats_df.to_csv(outdir / "features_used.csv", index=False)

    # Matrix X
    X = feats_df[feature_cols].values.astype(float)
    events = feats_df["event"].astype(str).tolist()
    n = len(events)

    # 4) Standardize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 5) PCA (2D)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    pca_df = pd.DataFrame(
        {
            "event": events,
            "pc1": Z[:, 0],
            "pc2": Z[:, 1],
        }
    )
    pca_df.to_csv(outdir / "pca_projection.csv", index=False)

    # 6) KMeans clustering
    from sklearn.cluster import KMeans

    k = args.k if args.k and args.k > 0 else _auto_k(n)
    k = max(2, min(k, n))  # clamp
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = km.fit_predict(Xs)

    lab_df = pd.DataFrame({"event": events, "cluster": labels})
    lab_df.to_csv(outdir / "clustering_labels.csv", index=False)

    # 7) Summary
    summary = {
        "n_events": n,
        "features": feature_cols,
        "kmeans_k": int(k),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "clusters": {
            str(int(c)): lab_df[lab_df["cluster"] == c]["event"].tolist()
            for c in sorted(lab_df["cluster"].unique())
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    (outdir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 8) Plots
    if not args.no_plots:
        import matplotlib.pyplot as plt

        # --- Paper-ready label handling ---
        # Use short labels (E1, E2, ...) in plots to avoid truncation/overlap,
        # and save a mapping table for the paper/supplement.
        short_labels = [f"E{i+1}" for i in range(n)]
        label_map_df = pd.DataFrame({"short_label": short_labels, "event": events})
        label_map_df.to_csv(outdir / "label_map.csv", index=False)

        # PCA scatter (paper-ready)
        plt.figure(figsize=(8.5, 6.0))
        for c in sorted(np.unique(labels)):
            idx = labels == c
            plt.scatter(Z[idx, 0], Z[idx, 1], label=f"cluster {int(c)}")

        # Annotate with short labels to keep the figure clean
        for i, lab in enumerate(short_labels):
            plt.annotate(
                lab,
                (Z[i, 0], Z[i, 1]),
                fontsize=10,
                xytext=(4, 4),
                textcoords="offset points",
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Inter-event clustering (PCA + KMeans)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / "pca_clusters.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Dendrogram (paper-ready; optional)
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram

            L = linkage(Xs, method="ward")

            # Large bottom margin + tight bbox to prevent label clipping
            plt.figure(figsize=(12.0, 6.5))
            dendrogram(
                L,
                labels=short_labels,
                leaf_rotation=0,
                leaf_font_size=11,
            )
            plt.title("Hierarchical clustering (Ward)")
            plt.xlabel("Event")
            plt.ylabel("Ward distance")

            # Add extra space at bottom for labels
            plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])
            plt.savefig(outdir / "dendrogram.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            # If scipy not available, skip silently
            pass

    print("[OK] Clustering completado.")
    print(f"[OK] Output dir: {outdir}")
    print(f"[OK] Events: {n} | Features: {len(feature_cols)} | KMeans k={k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
