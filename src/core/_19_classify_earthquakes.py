#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_classify_earthquakes.py
=========================

ES:
- Genera *clustering jerárquico (dendrograma)* y una tabla de "clases" a nivel evento
  usando las métricas agregadas ya producidas por el pipeline (ELI/Comp/Sync y, si existe,
  explosion_index).
- Está pensado para que NO se quede colgado: no abre ventanas (plt.show), guarda PNG/PDF
  en resultados/_summary/.

EN:
- Builds hierarchical clustering (dendrogram) + event-level class table from already
  computed pipeline summaries (ELI/Comp/Sync and, if present, explosion_index).
- Designed to never "hang": no GUI windows; only saves figures to resultados/_summary/.

Uso típico / Typical:
    python src/core/19_classify_earthquakes.py

Opcional:
    python src/core/19_classify_earthquakes.py --root <ruta_proyecto> <event1> <event2> ...

Notas:
- Si no pasas eventos, intentará inferirlos desde resultados/_summary/eli_ranked.txt o
  escaneando resultados/*/ (carpetas de eventos).
"""

from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# matplotlib is used only for saving (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist, squareform
except Exception as e:
    raise SystemExit(
        "ERROR: scipy es requerido para dendrograma/clustering.\n"
        "Instala en tu env tamc:  pip install scipy\n"
        f"Detalle: {e}"
    )

# -----------------------------
# Helpers
# -----------------------------

def infer_root(passed: Optional[str]) -> Path:
    if passed:
        return Path(passed).expanduser().resolve()
    # script lives at <root>/src/core/19_classify_earthquakes.py
    return Path(__file__).resolve().parents[2]

def read_explosion_index(summary_dir: Path) -> Dict[str, float]:
    """
    Returns mapping event->explosion_index if explosion_index.csv exists.
    Expected columns: event, explosion_index (tolerant to extra cols).
    """
    f = summary_dir / "explosion_index.csv"
    if not f.exists():
        return {}
    import csv
    out: Dict[str, float] = {}
    with f.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        cols = [c.strip().lower() for c in reader.fieldnames or []]
        # best-effort column names
        ev_col = None
        xi_col = None
        for c in reader.fieldnames or []:
            cl = c.strip().lower()
            if cl in ("event", "event_id", "name"):
                ev_col = c
            if cl in ("explosion_index", "xi", "x_i"):
                xi_col = c
        for row in reader:
            if not ev_col:
                # fallback: first column
                ev_col = reader.fieldnames[0]
            if not xi_col:
                # fallback: try find any numeric-ish column named like explosion
                for c in reader.fieldnames:
                    if "explosion" in c.lower():
                        xi_col = c
                        break
            try:
                ev = (row.get(ev_col) or "").strip()
                if not ev:
                    continue
                xi = float(row.get(xi_col, "").strip())
                out[ev] = xi
            except Exception:
                continue
    return out

def read_eli_ranked(summary_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Parses resultados/_summary/eli_ranked.txt (the table you showed).
    Returns mapping event -> {"ELI":..., "Comp":..., "Sync":...}
    """
    f = summary_dir / "eli_ranked.txt"
    if not f.exists():
        return {}
    txt = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: Dict[str, Dict[str, float]] = {}
    # Lines look like:
    # 1   0.521   0.043   1.000  SpaceX Starship IFT-2 ...
    # Robust parser:
    for line in txt:
        line = line.strip()
        if not line or line.startswith("rank") or set(line) == {"-"}:
            continue
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.*)$", line)
        if not m:
            continue
        _, eli, comp, sync, event = m.groups()
        event = event.strip()
        try:
            out[event] = {"ELI": float(eli), "Comp": float(comp), "Sync": float(sync)}
        except Exception:
            continue
    return out

def infer_events(root: Path, eli_map: Dict[str, Dict[str, float]]) -> List[str]:
    if eli_map:
        return list(eli_map.keys())
    # fallback: scan resultados/ for event folders (excluding _summary and analysis)
    resultados = root / "resultados"
    if not resultados.exists():
        return []
    events = []
    for p in resultados.iterdir():
        if p.is_dir() and p.name not in ("_summary", "analysis"):
            events.append(p.name)
    return sorted(events)

def normalize_features(X: np.ndarray) -> np.ndarray:
    # z-score with safe epsilon
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", type=str, default=None, help="Ruta al root del proyecto (opcional).")
    ap.add_argument("--k", type=int, default=2, help="Número de clusters para corte (por defecto 2).")
    ap.add_argument("--metric", type=str, default="euclidean",
                    choices=["euclidean", "cityblock", "cosine"],
                    help="Métrica de distancia para clustering.")
    ap.add_argument("--linkage", type=str, default="ward",
                    choices=["ward", "average", "complete", "single"],
                    help="Método de linkage.")
    ap.add_argument("events", nargs="*", help="Lista de eventos. Si vacío, se infiere automáticamente.")
    args = ap.parse_args()

    root = infer_root(args.root)
    summary_dir = root / "resultados" / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    eli_map = read_eli_ranked(summary_dir)
    xi_map = read_explosion_index(summary_dir)

    events = args.events if args.events else infer_events(root, eli_map)
    if not events:
        raise SystemExit(
            "ERROR: No pude inferir eventos.\n"
            "Pasa una lista explícita o asegúrate de tener resultados/_summary/eli_ranked.txt "
            "o carpetas en resultados/."
        )

    # Build feature matrix
    feats = []
    missing = []
    for ev in events:
        row = []
        # ELI/Comp/Sync if available; else NaN
        m = eli_map.get(ev, None)
        if m is None:
            # Sometimes eli_ranked.txt uses spaces; your folder names use underscores.
            # Try a relaxed match:
            m2 = None
            for k in eli_map.keys():
                if k.replace(" ", "_") == ev or k.replace("_", " ") == ev:
                    m2 = eli_map[k]
                    break
            m = m2
        if m is None:
            missing.append(ev)
            row += [math.nan, math.nan, math.nan]
        else:
            row += [m.get("ELI", math.nan), m.get("Comp", math.nan), m.get("Sync", math.nan)]
        # explosion index if available; else NaN
        xi = xi_map.get(ev, math.nan)
        if math.isnan(xi):
            # relaxed match
            for k in xi_map.keys():
                if k.replace(" ", "_") == ev or k.replace("_", " ") == ev:
                    xi = xi_map[k]
                    break
        row += [xi]
        feats.append(row)

    X = np.array(feats, dtype=float)

    # If explosion_index is entirely missing, drop that column
    if np.all(np.isnan(X[:, -1])):
        X = X[:, :3]

    # Replace NaNs by column means (best effort)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_means, inds[1])

    Xn = normalize_features(X)

    # Distance + linkage
    if args.linkage == "ward" and args.metric != "euclidean":
        print("[WARN] 'ward' requiere euclidean; forzando metric=euclidean.")
        metric = "euclidean"
    else:
        metric = args.metric

    D = pdist(Xn, metric=metric)
    Z = linkage(D, method=args.linkage)

    # Flat clusters
    k = max(2, int(args.k))
    labels = fcluster(Z, t=k, criterion="maxclust")

    # Save dendrogram
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    dendrogram(
        Z,
        labels=[e[:60] + ("…" if len(e) > 60 else "") for e in events],
        leaf_rotation=90,
        ax=ax,
    )
    ax.set_title(f"Event clustering (k={k}) — features: " + ("ELI/Comp/Sync" + ("+XI" if X.shape[1] == 4 else "")))
    ax.set_ylabel("Distance")
    fig.tight_layout()
    out_png = summary_dir / "event_dendrogram.png"
    out_pdf = summary_dir / "event_dendrogram.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    # Save table
    import csv
    out_csv = summary_dir / "earthquake_classification.csv"
    headers = ["event", "cluster", "ELI", "Comp", "Sync", "explosion_index"]
    # Build back values in original scale (from eli_map & xi_map, with relaxed match)
    def get_vals(ev: str) -> Tuple[float, float, float, float]:
        m = eli_map.get(ev, None)
        if m is None:
            for k0 in eli_map.keys():
                if k0.replace(" ", "_") == ev or k0.replace("_", " ") == ev:
                    m = eli_map[k0]
                    break
        eli = m.get("ELI", math.nan) if m else math.nan
        comp = m.get("Comp", math.nan) if m else math.nan
        sync = m.get("Sync", math.nan) if m else math.nan
        xi = xi_map.get(ev, math.nan)
        if math.isnan(xi):
            for k0 in xi_map.keys():
                if k0.replace(" ", "_") == ev or k0.replace("_", " ") == ev:
                    xi = xi_map[k0]
                    break
        return eli, comp, sync, xi

    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(headers)
        for ev, lab in sorted(zip(events, labels), key=lambda t: (t[1], t[0])):
            eli, comp, sync, xi = get_vals(ev)
            w.writerow([ev, int(lab), eli, comp, sync, xi])

    print("\n[OK] Dendrogram:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print("[OK] Classification table:")
    print(f"  {out_csv}")

    if missing:
        print("\n[WARN] No encontré ELI/Comp/Sync en eli_ranked.txt para estos eventos (se imputaron medias):")
        for ev in missing:
            print("  -", ev)

if __name__ == "__main__":
    main()
