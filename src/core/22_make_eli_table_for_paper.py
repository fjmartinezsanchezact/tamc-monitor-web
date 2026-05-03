#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0001_22_make_eli_table_for_paper.py

Paper-ready LaTeX tables for the Explosion-likeness Index (ELI) produced by
21_explosion_index.py.

Fixes the "super long names" issue by:
  - generating a compact "Event" label for the main-paper table
  - producing an optional Supplementary table with the full event_id in monospace
    and with safe line-break hints

Inputs (relative to repo root):
  resultados/_summary/explosion_index.csv

Outputs:
  resultados/_summary/paper_table_explosion_index_main.tex
  resultados/_summary/paper_table_explosion_index_supp.tex
  resultados/_summary/paper_table_explosion_index.csv  (cleaned + formatted)
"""

from __future__ import annotations

import csv
import math
import re


def make_site_year_label(event_id: str) -> str:
    """
    Convert an event_id like:
      '48_km_W_of_Illapel_Chile_M8.3_20150916_225432'
    into:
      '48 km W of Illapel Chile (2015)'
    Best-effort parsing; falls back gracefully if parsing fails.
    """
    if not event_id:
        return "UNKNOWN"
    s = event_id.replace("_", " ")

    # Extract year from YYYYMMDD if present
    m = re.search(r"\b((?:19|20)\d{2})(\d{2})(\d{2})\b", s)  # YYYYMMDD
    year = m.group(1) if m else None

    # Remove magnitude tokens like M8.3, Mw6.3
    s = re.sub(r"\bM[w]?\s*\d+(?:\.\d+)?\b", "", s).strip()
    s = re.sub(r"\bM\s*\d+(?:\.\d+)?\b", "", s).strip()

    # Truncate everything after the date token
    if m:
        date_token = m.group(0)
        s = s.split(date_token)[0].strip()

    # Drop common suffixes
    s = re.sub(r"\b(mainshock|control)\b", "", s, flags=re.IGNORECASE).strip()

    # Clean spaces
    s = re.sub(r"\s+", " ", s).strip()

    if year:
        return f"{s} ({year})"
    return s

from pathlib import Path


def repo_root() -> Path:
    # This file is expected to live in src/core/
    # repo root is 2 levels up: src/core -> src -> <root>
    return Path(__file__).resolve().parents[2]


def _latex_escape(s: str) -> str:
    # Minimal escaping for LaTeX tables
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _fmt(x: str, nd: int = 3) -> str:
    if x is None:
        return "NA"
    xs = str(x).strip()
    if xs == "" or xs.lower() in {"na", "nan"}:
        return "NA"
    try:
        v = float(xs)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return xs


def _insert_break_hints(s: str) -> str:
    """
    Add LaTeX-friendly break hints to long identifiers.
    We keep underscores escaped, but also allow breaking after separators.
    """
    # Work in raw string first; we will escape afterwards.
    # Insert '\allowbreak' after common separators.
    s = re.sub(r"([_/.-])", r"\1\\allowbreak{}", s)
    return s


def short_event_label(event_id: str) -> str:
    """
    Produce a compact, human-readable event label for the main paper.
    Tries keyword mapping first; falls back to extracting a year; then truncates.
    """
    s = (event_id or "").lower()

    # Keyword-based mapping (robust against long pipeline-generated ids)
    mapping = [
        (("dprk", "north_korea", "najibeagan"), "DPRK 2017"),
        (("chelyabinsk",), "Chelyabinsk 2013"),
        (("beirut", "lebanon"), "Beirut 2020"),
        (("brumadinho", "tailings", "minas_gerais"), "Brumadinho 2019"),
    ]
    for keys, label in mapping:
        if any(k in s for k in keys):
            return label

    # Try to infer year from an 8-digit date like 20170903 or 20130215
    m = re.search(r"(19|20)\d{2}", event_id or "")
    if m:
        return f"Event {m.group(0)}"

    # Fallback: truncated id
    ev = event_id or "Event"
    return (ev[:32] + "…") if len(ev) > 33 else ev


def main() -> None:
    root = repo_root()
    in_csv = root / "resultados" / "_summary" / "explosion_index.csv"
    out_dir = root / "resultados" / "_summary"
    out_tex_main = out_dir / "paper_table_explosion_index_main.tex"
    out_tex_supp = out_dir / "paper_table_explosion_index_supp.tex"
    out_csv = out_dir / "paper_table_explosion_index.csv"

    if not in_csv.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {in_csv}\n"
            "Run 21_explosion_index.py first to generate resultados/_summary/explosion_index.csv"
        )

    rows = []
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Sort by ELI descending (numeric when possible)
    def keyfun(r):
        try:
            return float(r.get("eli", "nan"))
        except Exception:
            return float("nan")

    rows.sort(key=lambda r: (-(keyfun(r)) if not math.isnan(keyfun(r)) else 1e9))

    # Clean + format rows for CSV export
    cleaned = []
    for r in rows:
        ev_id = r.get("event_id", "")
        cleaned.append(
            {
                "event_id": ev_id,
                "event_label": short_event_label(ev_id),
                "phase": r.get("phase", ""),
                "eli": _fmt(r.get("eli", "")),
                "compactness_near_t0": _fmt(r.get("compactness_near_t0", "")),
                "sync_alignment_t0": _fmt(r.get("sync_alignment_t0", "")),
                "precursor_strength": _fmt(r.get("precursor_strength", "")),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "event_label",
                "phase",
                "eli",
                "compactness_near_t0",
                "sync_alignment_t0",
                "precursor_strength",
            ],
        )
        w.writeheader()
        w.writerows(cleaned)

    # ---------------------------
    # MAIN PAPER TABLE (compact)
    # ---------------------------
    # Keep it tight: Event label + ELI/Comp/Sync (Prec optional; keep it if you want).
    header_main = [
        r"\begin{tabular}{l r r r}",
        r"\hline",
        r"\textbf{Event} & \textbf{ELI} & \textbf{Comp.} & \textbf{Sync.} \\",
        r"\hline",
    ]
    body_main = []
    for r in cleaned:
        ev = _latex_escape(r["event_label"])
        eli = r["eli"]
        comp = r["compactness_near_t0"]
        syn = r["sync_alignment_t0"]
        body_main.append(f"{ev} & {eli} & {comp} & {syn} \\\\")
    footer_main = [r"\hline", r"\end{tabular}"]
    out_tex_main.write_text("\n".join(header_main + body_main + footer_main) + "\n", encoding="utf-8")

    # ---------------------------------
    # SUPPLEMENTARY TABLE (full event_id)
    # ---------------------------------
    # Full identifiers, but with break hints and monospace; include Prec.
    header_supp = [
        r"\begin{tabular}{p{0.58\linewidth} l r r r r}",
        r"\hline",
        r"\textbf{Event ID} & \textbf{Phase} & \textbf{ELI} & \textbf{Comp.} & \textbf{Sync.} & \textbf{Prec.} \\",
        r"\hline",
    ]
    body_supp = []
    for r in cleaned:
        ev_raw = r["event_id"]
        ev_raw = _insert_break_hints(ev_raw)
        ev = _latex_escape(ev_raw)
        ph = _latex_escape(r["phase"])
        eli = r["eli"]
        comp = r["compactness_near_t0"]
        syn = r["sync_alignment_t0"]
        pre = r["precursor_strength"]
        ev_tex = r"{\ttfamily " + ev + "}"
        body_supp.append(f"{ev_tex} & {ph} & {eli} & {comp} & {syn} & {pre} \\\\")
    footer_supp = [r"\hline", r"\end{tabular}"]
    out_tex_supp.write_text("\n".join(header_supp + body_supp + footer_supp) + "\n", encoding="utf-8")

    print("Wrote:", out_tex_main)
    print("Wrote:", out_tex_supp)
    print("Wrote:", out_csv)

    # ---------------------------
    # CONSOLE LISTING (ranked ELI)
    # ---------------------------
    # Build a clean, readable label and print a ranked table to stdout.
    # Also write a text file for quick copy/paste into notes.
    out_txt = out_dir / "eli_ranked.txt"

    # cleaned is already sorted by ELI desc in this script
    lines = []
    header = f"{'rank':>4}  {'ELI':>6}  {'Comp':>6}  {'Sync':>6}  Event"
    sep = "-" * (len(header) + 10)
    lines.append(header)
    lines.append(sep)

    rank = 0
    for r in cleaned:
        eli = r.get("eli", "NA")
        comp = r.get("compactness_near_t0", "NA")
        sync = r.get("sync_alignment_t0", "NA")
        # Skip rows with missing ELI
        if eli in (None, "", "NA"):
            continue
        rank += 1
        event_id = r.get("event_id", "")
        label = make_site_year_label(event_id)
        lines.append(f"{rank:4d}  {eli:>6}  {comp:>6}  {sync:>6}  {label}")

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out_txt.write_text(text + "\n", encoding="utf-8")
    print("Wrote:", out_txt)


if __name__ == "__main__":
    main()
