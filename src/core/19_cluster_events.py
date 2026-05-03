#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_cluster_events.py
Earthquake clustering based on REAL inter-event summary features.
FIXED: plotting code moved inside main(), no stray returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_real_summary(root: Path) -> pd.DataFrame:
    path = root / "resultados" / "null_tests_13_14" / "analysis" / "real_multi_phase_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV REAL:\n{path}")
    df = pd.read_csv(path)
    if "event" not in df.columns:
        raise ValueError("El CSV no contiene columna 'event'")
    return df


def _filter_events(df: pd.DataFrame, selected_events):
    if not selected_events:
        return df.copy()
    sel = set(selected_events)
    return df[df["event"].astype(str).isin(sel)].copy()


def _prepare_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    numeric_cols = []
    for c in df.columns:
        if c == "event":
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() >= 2:
            df[c] = coerced
            numeric_cols.append(c)

    if len(numeric_cols) < 2:
        raise ValueError("No hay suficientes columnas numéricas")

    df["_n_valid"] = df[numeric_cols].notna().sum(axis=1)
    df = df[df["_n_valid"] >= 2].drop(columns=["_n_valid"])

    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    return df[["event"] + numeric_cols], numeric_cols


def _auto_k(n):
    return 2 if n <= 8 else 3


def _short_event_label(ev: str) -> str:
    years = re.findall(r"(?:19|20)\d{2}", ev)
    year = years[-1] if years else ""
    base = ev.split("__")[0].replace("_manual", "")
    words = [w for w in base.split("_") if not w.isdigit()]
    site = " ".join(words[:5])
    return f"{site} ({year})" if year else site


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("events", nargs="*")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / "resultados" / "earthquake_clustering"
    _safe_mkdir(outdir)

    df = _read_real_summary(root)
    df = _filter_events(df, args.events)

    feats_df, feature_cols = _prepare_feature_matrix(df)
    feats_df.to_csv(outdir / "features_used.csv", index=False)

    X = feats_df[feature_cols].values
    events = feats_df["event"].astype(str).tolist()
    n = len(events)
    ids = [f"{i+1:02d}" for i in range(n)]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(Xs)

    k = args.k if args.k > 0 else _auto_k(n)
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Xs)

    pd.DataFrame({
        "id": ids, "event": events, "cluster": labels,
        "pc1": Z[:,0], "pc2": Z[:,1]
    }).to_csv(outdir / "event_id_mapping.csv", index=False)

    if not args.no_plots:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram

        table_data = [[ids[i], _short_event_label(events[i])] for i in range(n)]

        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios":[4,2]})
        for c in sorted(set(labels)):
            m = labels == c
            ax.scatter(Z[m,0], Z[m,1], label=f"cluster {c}")
        for i,eid in enumerate(ids):
            ax.annotate(eid, (Z[i,0], Z[i,1]), fontsize=9)
        ax.legend(); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=table_data, colLabels=["ID","Evento"], loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.2)
        fig.tight_layout()
        fig.savefig(outdir / "pca_clusters.png", dpi=150)
        plt.close(fig)

        L = linkage(Xs, method="ward")
        fig, (ax, ax_tbl) = plt.subplots(2,1, figsize=(10, max(6,0.3*n+4)))
        dendrogram(L, labels=ids, ax=ax)
        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=table_data, colLabels=["ID","Evento"], loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.2)
        fig.tight_layout()
        fig.savefig(outdir / "dendrogram.png", dpi=150)
        plt.close(fig)

    print("[OK] Clustering completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
