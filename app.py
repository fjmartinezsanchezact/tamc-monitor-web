from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_RESULTADOS_DIR = "web_data/latest"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
RAW_DATA_DIR_NAMES = {"raw", "mseed", "waveforms", "data"}

SEISMIC_DOI = "https://doi.org/10.5281/zenodo.19665949"
VOLCANIC_DOI = "https://doi.org/10.5281/zenodo.18525626"
EXTREME_EVENT_DOI = "https://doi.org/10.5281/zenodo.18649274"

GRAPH_SLOTS = {
    "Multistation synchrony": {
        "keywords": ["sync_multistation", "sync", "synchrony", "coherence", "coherencia"],
        "description": (
            "Network-level temporal synchrony across stations. This is one of the core "
            "outputs of the framework: it measures collective behaviour, not only amplitude."
        ),
    },
    "Extreme anomaly distribution": {
        "keywords": ["extremos", "extreme", "peaks", "valleys"],
        "description": (
            "Distribution of extreme anomaly values across the analysis window. "
            "This panel highlights when the strongest departures from baseline occur."
        ),
    },
    "Mean anomaly and susceptibility": {
        "keywords": ["medio", "mean", "susceptibility", "x(t)", "susceptibilidad"],
        "description": (
            "Temporal evolution of mean anomaly and susceptibility-like behaviour, useful "
            "for identifying structured increases and post-event decay."
        ),
    },
    "Station-resolved z-scores": {
        "keywords": ["zscore", "z_score", "rot", "24h", "station", "z"],
        "description": (
            "Station-level standardized activity across the full window. Simultaneous "
            "excursions across multiple stations support a distributed network response."
        ),
    },
    "Anomaly vs. synthetic tidal forcing": {
        "keywords": ["forzantes", "marea", "tide", "forcing"],
        "description": (
            "Comparison between anomaly metrics and synthetic tidal forcing. This helps "
            "assess whether the observed structure resembles a smooth external modulation "
            "or a sharper seismic response."
        ),
    },
}

ZONE_LABELS = {
    "torremolinos": "Torremolinos / Alboran Sea, Spain",
    "miyako": "Miyako, Japan",
    "maule": "Maule, Chile",
    "san_andreas": "San Andreas Fault, California, USA",
    "sanandreas": "San Andreas Fault, California, USA",
    "cascadia": "Cascadia Subduction Zone, USA–Canada",
    "aegean": "Aegean Sea, Greece–Turkey",
    "hikurangi": "Hikurangi Subduction Zone, New Zealand",
    "yellowstone": "Yellowstone, Wyoming, USA",
    "iceland_reykjanes": "Reykjanes Peninsula, Iceland",
    "reykjanes": "Reykjanes Peninsula, Iceland",
    "la_palma": "La Palma, Canary Islands, Spain",
    "lapalma": "La Palma, Canary Islands, Spain",
    "etna": "Etna, Italy",
    "stromboli": "Stromboli, Italy",
    "kilauea": "Kīlauea, Hawaiʻi, USA",
    "hawaii": "Kīlauea, Hawaiʻi, USA",
    "fuego": "Fuego, Guatemala",
    "popocatepetl": "Popocatépetl, Mexico",
    "popocat": "Popocatépetl, Mexico",
}

ZONE_TIMEZONES = {
    "torremolinos": "Europe/Madrid",
    "miyako": "Asia/Tokyo",
    "maule": "America/Santiago",
    "san_andreas": "America/Los_Angeles",
    "sanandreas": "America/Los_Angeles",
    "cascadia": "America/Los_Angeles",
    "aegean": "Europe/Athens",
    "hikurangi": "Pacific/Auckland",
    "yellowstone": "America/Denver",
    "iceland_reykjanes": "Atlantic/Reykjavik",
    "reykjanes": "Atlantic/Reykjavik",
    "la_palma": "Atlantic/Canary",
    "lapalma": "Atlantic/Canary",
    "etna": "Europe/Rome",
    "stromboli": "Europe/Rome",
    "kilauea": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",
    "fuego": "America/Guatemala",
    "popocatepetl": "America/Mexico_City",
    "popocat": "America/Mexico_City",
}

VOLCANIC_KEYWORDS = [
    "volcan", "volcano", "volcanic", "eruption", "erupcion", "erupción",
    "reykjanes", "iceland", "la_palma", "lapalma", "yellowstone", "etna", "stromboli",
    "kilauea", "hawaii", "fuego", "popocatepetl", "popocat"
]

EARTHQUAKE_KEYWORDS = [
    "earthquake", "sismo", "terremoto", "miyako", "maule", "torremolinos", "alboran", "japan", "chile"
]

FAULT_SUBDUCTION_KEYWORDS = [
    "fault", "subduction", "san_andreas", "sanandreas", "cascadia", "hikurangi", "aegean", "greece", "turkey"
]

SEISMIC_KEYWORDS = EARTHQUAKE_KEYWORDS + FAULT_SUBDUCTION_KEYWORDS

LAYER_OPTIONS = [
    "All regions",
    "Earthquake monitoring",
    "Volcanic monitoring",
    "Fault & subduction zones",
]

REGION_CONTEXT = {
    "reykjanes": (
        "Why this region matters",
        "Active magmatic intrusion system with continuous seismic and deformation signals. Useful for testing whether multistation coherence captures distributed subsurface reorganization."
    ),
    "iceland_reykjanes": (
        "Why this region matters",
        "Active magmatic intrusion system with continuous seismic and deformation signals. Useful for testing whether multistation coherence captures distributed subsurface reorganization."
    ),
    "cascadia": (
        "Why this region matters",
        "Subduction zone known for slow slip events and long-duration weak signals. Relevant for testing temporally extended collective behaviour."
    ),
    "etna": (
        "Why this region matters",
        "Highly active volcanic system with frequent eruptive episodes. Provides a useful environment to compare coherent multistation responses."
    ),
    "stromboli": (
        "Why this region matters",
        "Open-conduit volcanic system with persistent activity. Suitable for analyzing continuous network-level dynamics."
    ),
    "kilauea": (
        "Why this region matters",
        "Well-instrumented volcanic system with diverse activity patterns. Acts as a benchmark for comparison with established monitoring frameworks."
    ),
    "hawaii": (
        "Why this region matters",
        "Well-instrumented volcanic system with diverse activity patterns. Acts as a benchmark for comparison with established monitoring frameworks."
    ),
    "popocatepetl": (
        "Why this region matters",
        "Highly active volcanic system with frequent explosive activity. Useful for evaluating strong localized multistation responses."
    ),
    "popocat": (
        "Why this region matters",
        "Highly active volcanic system with frequent explosive activity. Useful for evaluating strong localized multistation responses."
    ),
    "fuego": (
        "Why this region matters",
        "Continuously active volcanic system with intermittent explosive behaviour. A challenging environment for multistation statistical consistency."
    ),
    "la_palma": (
        "Why this region matters",
        "Volcanic island system with recent eruptive history. Relevant for testing reproducibility across well-documented volcanic crises."
    ),
    "lapalma": (
        "Why this region matters",
        "Volcanic island system with recent eruptive history. Relevant for testing reproducibility across well-documented volcanic crises."
    ),
    "san_andreas": (
        "Why this region matters",
        "Major strike-slip fault system with complex seismic behaviour. Included to assess non-volcanic collective dynamics."
    ),
    "sanandreas": (
        "Why this region matters",
        "Major strike-slip fault system with complex seismic behaviour. Included to assess non-volcanic collective dynamics."
    ),
    "hikurangi": (
        "Why this region matters",
        "Subduction zone with documented slow slip phenomena. Relevant for detecting distributed and low-amplitude network structure."
    ),
    "aegean": (
        "Why this region matters",
        "Tectonically complex region with mixed seismic regimes. Useful for testing robustness across heterogeneous systems."
    ),
    "miyako": (
        "Why this region matters",
        "Dense seismic network in a highly active tectonic region. Provides high-quality data for multistation analysis."
    ),
    "maule": (
        "Why this region matters",
        "Large-scale subduction system with well-documented seismic behaviour. Suitable for comparison across tectonic regimes."
    ),
    "torremolinos": (
        "Why this region matters",
        "Local reference region used for continuous monitoring under identical conditions. Acts as a baseline for comparison with other regions."
    ),
    "alboran": (
        "Why this region matters",
        "Local southern-Spain seismic reference region processed under identical monitoring conditions. Useful as a regional baseline."
    ),
    "yellowstone": (
        "Why this region matters",
        "Volcanic and hydrothermal system with strong public and scientific interest. Useful for comparing volcanic-region network behaviour."
    ),
}


# ============================================================
# HELPERS
# ============================================================
def get_resultados_dir() -> Path | None:
    if DEFAULT_RESULTADOS_DIR:
        p = Path(DEFAULT_RESULTADOS_DIR).expanduser().resolve()
        return p if p.exists() else None
    return None


def list_zone_dirs(resultados_dir: Path) -> List[Path]:
    return sorted([p for p in resultados_dir.iterdir() if p.is_dir()], key=lambda x: x.name.lower())


def find_images_recursive(zone_dir: Path) -> List[Path]:
    images: List[Path] = []
    for p in zone_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return sorted(images, key=lambda x: (x.parent.as_posix().lower(), x.name.lower()))


def score_image_for_slot(filename: str, keywords: List[str]) -> int:
    name = filename.lower()
    return sum(1 for kw in keywords if kw.lower() in name)


def assign_images_to_slots(images: List[Path]) -> Dict[str, Path | None]:
    remaining = images[:]
    assigned: Dict[str, Path | None] = {slot: None for slot in GRAPH_SLOTS}

    for slot, cfg in GRAPH_SLOTS.items():
        best_img = None
        best_score = 0
        for img in remaining:
            score = score_image_for_slot(img.name, cfg["keywords"])
            if score > best_score:
                best_score = score
                best_img = img
        if best_img is not None and best_score > 0:
            assigned[slot] = best_img
            remaining.remove(best_img)

    for slot in GRAPH_SLOTS:
        if assigned[slot] is None and remaining:
            assigned[slot] = remaining.pop(0)
    return assigned


def prettify_zone_name(raw_name: str) -> str:
    raw_lower = raw_name.lower()
    for key, label in ZONE_LABELS.items():
        if key in raw_lower:
            return label
    return raw_name.replace("__", " ").replace("_", " ").strip()


def classify_zone(raw_name: str) -> str:
    """Primary physical layer shown on cards and counters."""
    name = raw_name.lower()
    if any(k in name for k in VOLCANIC_KEYWORDS):
        return "Volcanic monitoring"
    if any(k in name for k in FAULT_SUBDUCTION_KEYWORDS):
        return "Fault & subduction zones"
    if any(k in name for k in EARTHQUAKE_KEYWORDS):
        return "Earthquake monitoring"
    return "Earthquake monitoring"


def zone_matches_layer(zone_dir: Path, layer: str) -> bool:
    if layer == "All regions":
        return True
    return classify_zone(zone_dir.name) == layer


def get_zone_timezone(raw_name: str) -> str | None:
    raw_lower = raw_name.lower()
    for key, tz_name in ZONE_TIMEZONES.items():
        if key in raw_lower:
            return tz_name
    return None


def extract_datetime_from_name(name: str) -> datetime | None:
    # Supports names ending in ..._YYYYMMDD_HHMMSS
    match = re.search(r"(20\d{6})[_-](\d{6})", name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def format_zone_datetime(raw_name: str) -> str:
    dt_utc = extract_datetime_from_name(raw_name)
    if dt_utc is None:
        return "Timestamp not detected"

    utc_str = dt_utc.strftime("%Y-%m-%d %H:%M UTC")
    tz_name = get_zone_timezone(raw_name)
    if tz_name and ZoneInfo is not None:
        try:
            local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
            return f"{utc_str} · {local_dt.strftime('%Y-%m-%d %H:%M')} local"
        except Exception:
            return utc_str
    return utc_str


def zone_sort_key(zone_dir: Path) -> str:
    dt = extract_datetime_from_name(zone_dir.name)
    if dt is None:
        return "00000000000000"
    return dt.strftime("%Y%m%d%H%M%S")


def zone_intro_text(raw_name: str) -> str:
    name = raw_name.lower()
    if "torremolinos" in name or "alboran" in name:
        return "Local southern-Spain reference region used for continuous fixed-pipeline monitoring."
    if "miyako" in name:
        return "Japanese seismic reference region used for fixed-window multistation monitoring."
    if "maule" in name:
        return "Chilean subduction reference region used for comparison across tectonic settings."
    if "san_andreas" in name or "sanandreas" in name:
        return "Major strike-slip fault system included for experimental fixed-window monitoring."
    if "cascadia" in name:
        return "Major subduction-zone system included for experimental multistation monitoring."
    if "aegean" in name or "greece" in name or "turkey" in name:
        return "Active eastern Mediterranean tectonic region monitored under the same fixed pipeline."
    if "hikurangi" in name:
        return "New Zealand subduction-zone system monitored for collective network structure."
    if "yellowstone" in name:
        return "Volcanic and hydrothermal region included for experimental collective-behaviour monitoring."
    if "reykjanes" in name or "iceland" in name:
        return "Active volcanic-seismic region suitable for continuous experimental monitoring."
    if "la_palma" in name or "lapalma" in name:
        return "Volcanic island region used to monitor collective multistation statistical structure."
    if "etna" in name:
        return "Highly active Italian volcanic system monitored for regional multistation structure."
    if "stromboli" in name:
        return "Open-conduit Italian volcanic system monitored for collective station behaviour."
    if "kilauea" in name or "hawaii" in name:
        return "Hawaiian volcanic system used as a high-interest global monitoring region."
    if "fuego" in name:
        return "Guatemalan volcanic system included in the global volcanic monitoring layer."
    if "popocatepetl" in name or "popocat" in name:
        return "Mexican volcanic system included for high-interest experimental monitoring."
    return "Experimental monitoring region processed with the same fixed TAMC–FRANJAMAR pipeline."


def zone_context(raw_name: str) -> Tuple[str, str] | None:
    """Short neutral scientific context for each monitored region."""
    raw_lower = raw_name.lower()
    for key, value in REGION_CONTEXT.items():
        if key in raw_lower:
            return value
    return None


def render_zone_context(raw_name: str) -> None:
    """Render a concise non-interpretive context block for a monitored region."""
    ctx = zone_context(raw_name)
    if not ctx:
        return
    title, body = ctx
    st.markdown(f"<div class='region-context'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)


def parse_station_code_from_mseed(filename: str) -> str | None:
    """Extract NETWORK.STATION from waveform filenames.

    Supported examples:
    - BK.SAO.00.BHE-BHN-BHZ_2026-04-22_RECENT.mseed  -> BK.SAO
    - CI.MPP.BHE-BHN-BHZ_2026-04-22_RECENT.mseed     -> CI.MPP
    """
    stem = Path(filename).stem

    match = re.match(r"^([A-Za-z0-9]+)\.([A-Za-z0-9_-]+)(?:\.|_)", stem)
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    parts = stem.split(".")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}.{parts[1]}"
    return None


def station_search_roots(zone_dir: Path) -> List[Path]:
    """Return possible folders where raw waveform files may live.

    The app may be pointed either at the output folder or at the data folder.
    This searches both the selected zone folder and common sibling layouts such as:
    - <zone>/mainshock/raw
    - <project>/data/<zone>/mainshock/raw
    - <project>/resultados/<zone> plus <project>/data/<zone>
    """
    zone_dir = zone_dir.resolve()
    roots: List[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.exists() and path not in roots:
            roots.append(path)

    add(zone_dir)

    for rel in [
        Path("raw"),
        Path("mainshock") / "raw",
        Path("mseed"),
        Path("waveforms"),
        Path("data"),
    ]:
        add(zone_dir / rel)

    possible_bases = [zone_dir.parent, zone_dir.parent.parent, Path.cwd(), Path.cwd().parent]
    for base in possible_bases:
        for data_name in ["data", "DATA", "Data"]:
            candidate = base / data_name / zone_dir.name
            add(candidate)
            add(candidate / "mainshock" / "raw")
            add(candidate / "raw")

    return roots


def find_station_files(zone_dir: Path) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for root in station_search_roots(zone_dir):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".mseed", ".miniseed", ".sac"}:
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    files.append(rp)
    return sorted(files, key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))


@st.cache_data(show_spinner=False)
def load_zone_images(zone_dir_str: str) -> Tuple[Dict[str, str | None], List[str]]:
    zone_dir = Path(zone_dir_str)
    images = find_images_recursive(zone_dir)
    assigned = assign_images_to_slots(images)
    return {k: str(v) if v else None for k, v in assigned.items()}, [str(p) for p in images]


def find_fixed_stations_json(zone_dir: Path) -> Path | None:
    """Locate fixed_stations.json for a zone.

    Preferred source for station metadata. It usually contains network, station,
    latitude, longitude and distance to the monitored reference point.
    """
    zone_dir = zone_dir.resolve()
    candidates: List[Path] = [
        zone_dir / "fixed_stations.json",
        zone_dir / "mainshock" / "fixed_stations.json",
    ]

    possible_bases = [zone_dir.parent, zone_dir.parent.parent, Path.cwd(), Path.cwd().parent]
    for base in possible_bases:
        for data_name in ["data", "DATA", "Data"]:
            candidates.append(base / data_name / zone_dir.name / "fixed_stations.json")
            candidates.append(base / data_name / zone_dir.name / "mainshock" / "fixed_stations.json")

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    for root in station_search_roots(zone_dir):
        hit = root / "fixed_stations.json"
        if hit.exists() and hit.is_file():
            return hit.resolve()
    return None


def load_zone_station_metadata(zone_dir_str: str) -> Tuple[List[str], List[Dict[str, object]], List[str], str]:
    """Return station codes, detailed station metadata, source files and source label.

    First tries fixed_stations.json because it contains coordinates and distances.
    Falls back to waveform filenames if metadata JSON is missing.
    """
    zone_dir = Path(zone_dir_str)
    fixed_json = find_fixed_stations_json(zone_dir)

    if fixed_json is not None:
        try:
            with open(fixed_json, "r", encoding="utf-8") as f:
                raw = json.load(f)

            details: List[Dict[str, object]] = []
            codes: List[str] = []
            for item in raw:
                net = str(item.get("net", "")).strip()
                sta = str(item.get("sta", "")).strip()
                loc = str(item.get("loc", "")).strip()
                if not net or not sta:
                    continue

                code = f"{net}.{sta}"
                codes.append(code)
                details.append({
                    "code": code,
                    "network": net,
                    "station": sta,
                    "location": loc if loc else "—",
                    "lat": item.get("stalat"),
                    "lon": item.get("stalon"),
                    "dist_km": item.get("dist_km"),
                })

            details = sorted(details, key=lambda x: str(x.get("code", "")))
            return sorted(set(codes)), details, [str(fixed_json)], "fixed_stations.json"
        except Exception:
            pass

    # Fallback: infer only station codes from waveform filenames.
    files = find_station_files(zone_dir)
    codes = []
    for f in files:
        code = parse_station_code_from_mseed(f.name)
        if code:
            codes.append(code)

    unique = sorted(set(codes))
    details = [
        {
            "code": code,
            "network": code.split(".")[0] if "." in code else code,
            "station": code.split(".")[1] if "." in code else "—",
            "location": "—",
            "lat": None,
            "lon": None,
            "dist_km": None,
        }
        for code in unique
    ]
    return unique, details, [str(f) for f in files], "waveform filenames"


def load_zone_stations(zone_dir_str: str) -> Tuple[List[str], List[str]]:
    """Backward-compatible wrapper used by older parts of the app."""
    stations, _details, source_files, _source_label = load_zone_station_metadata(zone_dir_str)
    return stations, source_files


def render_station_metadata(details: List[Dict[str, object]], source_label: str, source_files: List[str]) -> None:
    """Display station metadata in a compact, readable and auditable way."""
    if not details:
        st.warning("No station metadata was detected for this region.")
        return

    st.markdown(f"**Stations used ({len(details)}):**")
    st.caption(
        f"Station metadata source: `{source_label}` · "
        "metadata generated by the pipeline at run time for full reproducibility"
    )

    rows = []
    for d in details:
        lat = d.get("lat")
        lon = d.get("lon")
        dist = d.get("dist_km")
        rows.append({
            "Code": d.get("code", ""),
            "Network": d.get("network", ""),
            "Station": d.get("station", ""),
            "Loc": d.get("location", "—"),
            "Latitude": None if lat is None else round(float(lat), 4),
            "Longitude": None if lon is None else round(float(lon), 4),
            "Distance km": None if dist is None else round(float(dist), 1),
        })

    st.markdown("**Station metadata (network, location and geometry):**")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Coordinates and distances correspond to the fixed station set used for this run. "
        "All stations are selected automatically by the pipeline under identical criteria."
    )

    st.code("  ".join(str(d.get("code", "")) for d in details), language="text")

    with st.expander("Show station metadata source file(s)"):
        for f in source_files:
            st.write(Path(f).name)


def pick_hero(zone_dirs: List[Path]) -> Tuple[Path | None, str | None, str | None]:
    """Pick a stable hero figure.

    Priority is given to Reykjanes/Iceland because it is the intended scientific demo:
    an active volcanic-seismic system with continuous dynamics. If no Reykjanes output
    is available, the function falls back to a recent region with a usable image.
    """
    preferred_zones = [z for z in zone_dirs if any(k in z.name.lower() for k in ["reykjanes", "iceland"])]
    fallback_zones = zone_dirs[: min(8, len(zone_dirs))]
    search_order = preferred_zones + [z for z in fallback_zones if z not in preferred_zones]

    preferred_slots = [
        "Extreme anomaly distribution",
        "Multistation synchrony",
        "Mean anomaly and susceptibility",
        "Station-resolved z-scores",
        "Anomaly vs. synthetic tidal forcing",
    ]

    for z in search_order:
        assigned, _ = load_zone_images(str(z.resolve()))
        for slot in preferred_slots:
            img = assigned.get(slot)
            if img and Path(img).exists():
                return z, slot, img

    return None, None, None


def render_disclaimer() -> None:
    """Render the main scientific disclaimer and reproducibility statement."""
    html = """
<div class="disclaimer-card">
  <div class="badge-red">IMPORTANT NOTICE</div>
  <h2>Experimental research system</h2>
  <p>This platform is based on the <b>TAMC-FRANJAMAR</b> framework and is designed for the scientific exploration of collective multistation statistical behaviour.</p>

  <p><b>Core principle: identical processing across all regions</b></p>

  <p>All regions are processed using the exact same fixed pipeline, including:<br>
  • identical parameters<br>
  • identical preprocessing<br>
  • identical statistical thresholds<br><br>
  <b>No region-specific tuning, calibration, or manual adjustment is applied at any stage.</b></p>

  <p>This ensures full reproducibility and allows direct comparison of collective behaviour across earthquakes, volcanic systems, and major fault/subduction zones.</p>

  <p>It analyzes <b>temporal coherence and collective statistical structure across multiple stations</b>. It is not designed as a conventional amplitude-based event detector.</p>

  <p class="negative">
  It does <b>not</b> provide predictions, does <b>not</b> constitute an early warning system, and must <b>not</b> be used for operational, civil-protection, aviation, navigation, infrastructure, or safety decisions.
  </p>

  <hr style="opacity:0.2;">

  <p class="muted">
  Fully reproducible research framework - detailed documentation and datasets available in the <b>"DOI / Papers"</b> section.
  </p>

</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_header(zone_dirs: List[Path]) -> None:
    # Data-window timestamp inferred from the newest result folder name.
    # Expected pattern: ..._YYYYMMDD_HHMMSS
    latest_dt = extract_datetime_from_name(zone_dirs[0].name) if zone_dirs else None
    latest_txt = latest_dt.strftime("%Y-%m-%d %H:%M UTC") if latest_dt else "not detected"

    # System update timestamp: when this dashboard instance is being rendered/refreshed.
    # If the app is rebuilt/deployed every 12 h, this is the visible UTC update time.
    update_dt = datetime.now(timezone.utc)
    update_txt = update_dt.strftime("%Y-%m-%d %H:%M UTC")

    earthquake_count = sum(1 for z in zone_dirs if classify_zone(z.name) == "Earthquake monitoring")
    volcanic_count = sum(1 for z in zone_dirs if classify_zone(z.name) == "Volcanic monitoring")
    fault_count = sum(1 for z in zone_dirs if classify_zone(z.name) == "Fault & subduction zones")

    st.markdown(
        f"""
        <div class="top-hero">
            <div>
                <div class="eyebrow">EXPERIMENTAL MONITOR</div>
                <h1>Multistation coherence monitor</h1>
                <p>
                    Fixed-parameter monitoring of temporal coherence and collective statistical structure
                    across earthquake regions, volcanic systems, and major fault/subduction zones.
                </p>
            </div>
            <div class="metric-grid">
                <div class="metric"><span>{len(zone_dirs)}</span><small>regions</small></div>
                <div class="metric"><span>{earthquake_count}</span><small>earthquake</small></div>
                <div class="metric"><span>{volcanic_count}</span><small>volcanic</small></div>
                <div class="metric"><span>{fault_count}</span><small>fault/subduction</small></div>
            </div>
        </div>

        <div class="update-panel">
            <div class="update-box update-main">
                <span class="update-label">System update cycle</span>
                <b>Every 12 h</b>
            </div>
            <div class="update-box">
                <span class="update-label">Last update (UTC)</span>
                <b>{update_txt}</b>
            </div>
            <div class="update-box">
                <span class="update-label">Latest data window</span>
                <b>{latest_txt}</b>
            </div>
        </div>

        <div class="run-strip">
            All regions are processed under a strictly identical fixed pipeline · 
            no parameter tuning · no region-specific calibration · fully reproducible outputs · 
            24 h retrospective window · T−3 h delay · non-predictive monitoring<br><br>
            Seismic waveform data are automatically retrieved from public seismic networks via FDSN web services,
            including USGS and affiliated international providers. Station selection is performed automatically
            under identical criteria for all regions. Raw waveform data are handled in miniSEED-compatible form
            and processed through the same reproducible TAMC–FRANJAMAR workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_figure(zone_dirs: List[Path]) -> None:
    hero_zone, hero_slot, hero_img = pick_hero(zone_dirs)
    if not hero_zone or not hero_img:
        return
    st.markdown("### Featured monitoring output")
    st.markdown(
        f"<div class='hero-caption'><b>{prettify_zone_name(hero_zone.name)}</b> · {format_zone_datetime(hero_zone.name)} · {hero_slot}</div>",
        unsafe_allow_html=True,
    )
    hero_col_left, hero_col_mid, hero_col_right = st.columns([1, 2, 1])
    with hero_col_mid:
        st.image(hero_img, width=760)


def render_graph_thumbnail(slot: str, img: str | None, key_prefix: str) -> None:
    """Render one graph as a visible thumbnail with an expandable full-size view."""
    st.markdown(f"#### {slot}")
    st.caption(GRAPH_SLOTS[slot]["description"])

    if img and Path(img).exists():
        st.image(img, use_container_width=True)
        try:
            with st.popover(f"Open full-size: {slot}", use_container_width=True):
                st.image(img, caption=Path(img).name, use_container_width=True)
        except Exception:
            with st.expander(f"Open full-size: {slot}"):
                st.image(img, caption=Path(img).name, use_container_width=True)
    else:
        st.info("Image not found.")


def render_zone_card(zone_dir: Path) -> None:
    assigned, _ = load_zone_images(str(zone_dir.resolve()))
    stations, station_details, station_source_files, station_source_label = load_zone_station_metadata(str(zone_dir.resolve()))

    st.markdown("<div class='zone-card'>", unsafe_allow_html=True)
    st.markdown(f"#### {prettify_zone_name(zone_dir.name)}")
    st.caption(format_zone_datetime(zone_dir.name))
    st.write(zone_intro_text(zone_dir.name))
    render_zone_context(zone_dir.name)

    c1, c2, c3 = st.columns(3)
    c1.metric("Layer", classify_zone(zone_dir.name))
    c2.metric("Stations", len(stations) if stations else "—")
    c3.metric("Mode", "T−3 h")

    st.markdown("**Five core outputs**")
    graph_cols = st.columns(5)
    for idx, (slot, img) in enumerate(assigned.items()):
        with graph_cols[idx % 5]:
            st.markdown(f"<div class='graph-mini-title'>{slot}</div>", unsafe_allow_html=True)
            if img and Path(img).exists():
                st.image(img, use_container_width=True)
                try:
                    with st.popover("Expand", use_container_width=True):
                        st.markdown(f"### {prettify_zone_name(zone_dir.name)} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
                except Exception:
                    with st.expander("Expand"):
                        st.markdown(f"### {prettify_zone_name(zone_dir.name)} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
            else:
                st.info("Missing")

    st.markdown("</div>", unsafe_allow_html=True)

def render_zone_detail(zone_dir: Path, show_all: bool) -> None:
    display_name = prettify_zone_name(zone_dir.name)
    assigned, all_images = load_zone_images(str(zone_dir.resolve()))
    stations, station_details, station_source_files, station_source_label = load_zone_station_metadata(str(zone_dir.resolve()))

    st.markdown(f"## {display_name}")
    st.markdown(
        f"<div class='zone-meta'>{format_zone_datetime(zone_dir.name)} · {classify_zone(zone_dir.name)} · fixed TAMC–FRANJAMAR pipeline</div>",
        unsafe_allow_html=True,
    )
    st.write(zone_intro_text(zone_dir.name))

    tabs = st.tabs(["Overview", "Five graphs", "Stations"])

    with tabs[0]:
        st.markdown("### Overview")
        st.info(
            "This region is processed together with the other selected regions using the same fixed pipeline. "
            "The goal is to analyze temporal coherence and collective statistical structure across stations, "
            "not only raw signal amplitude."
        )
        st.markdown("### Five core outputs")
        cols = st.columns(5)
        for idx, (slot, img) in enumerate(assigned.items()):
            with cols[idx % 5]:
                st.markdown(f"<div class='graph-mini-title'>{slot}</div>", unsafe_allow_html=True)
                if img and Path(img).exists():
                    st.image(img, use_container_width=True)
                    try:
                        with st.popover("Expand", use_container_width=True):
                            st.markdown(f"### {slot}")
                            st.caption(GRAPH_SLOTS[slot]["description"])
                            st.image(img, caption=Path(img).name, use_container_width=True)
                            if station_details:
                                render_station_metadata(station_details, station_source_label, station_source_files)
                    except Exception:
                        with st.expander("Expand"):
                            st.markdown(f"### {slot}")
                            st.caption(GRAPH_SLOTS[slot]["description"])
                            st.image(img, caption=Path(img).name, use_container_width=True)
                            if station_details:
                                render_station_metadata(station_details, station_source_label, station_source_files)
                else:
                    st.info("Missing")

    with tabs[1]:
        st.markdown("### Five graph outputs")
        st.write(
            "Each monitored zone exposes the same five diagnostic panels. Click **Expand** under any panel "
            "to inspect it at full width together with the station list when available."
        )
        cols = st.columns(2)
        for idx, (slot, img) in enumerate(assigned.items()):
            with cols[idx % 2]:
                render_graph_thumbnail(slot, img, key_prefix=f"{zone_dir.name}_{idx}")

        if show_all:
            st.markdown("### All detected images")
            extra_cols = st.columns(3)
            for idx, img in enumerate(all_images):
                with extra_cols[idx % 3]:
                    st.image(img, caption=Path(img).name, use_container_width=True)

    with tabs[2]:
        st.markdown("### Stations used")
        st.write(
            "Station information is read preferentially from `fixed_stations.json`, which contains station "
            "coordinates and distance to the monitored reference point. If that file is not available, "
            "the dashboard falls back to inferring station codes from waveform filenames such as `mainshock/raw/*.mseed`."
        )
        if station_details:
            render_station_metadata(station_details, station_source_label, station_source_files)
        else:
            st.warning(
                "No station metadata or waveform files were detected. Add `fixed_stations.json` to each zone folder "
                "or keep waveform files under `mainshock/raw/*.mseed`."
            )

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070a12;
            --card: #111827;
            --card2: #0f172a;
            --line: #243044;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #38bdf8;
            --red: #f97373;
        }
        .stApp { background: radial-gradient(circle at top left, #111827 0, #070a12 38%, #05070d 100%); color: var(--text); }
        [data-testid="stSidebar"] { background: #0b1020; border-right: 1px solid #1f2937; }
        h1, h2, h3, h4 { color: #f8fafc; letter-spacing: -0.02em; }
        p, li, span, div { color: inherit; }
        a { color: #7dd3fc !important; text-decoration: none; }
        .top-hero {
            display: grid; grid-template-columns: 1.4fr 1fr; gap: 24px; align-items: center;
            padding: 28px; border: 1px solid var(--line); border-radius: 24px;
            background: linear-gradient(135deg, rgba(15,23,42,.95), rgba(2,6,23,.9));
            box-shadow: 0 18px 50px rgba(0,0,0,.35); margin-bottom: 14px;
        }
        .top-hero h1 { font-size: 2.4rem; margin: 0.1rem 0 0.6rem 0; }
        .top-hero p { color: var(--muted); font-size: 1.05rem; max-width: 850px; }
        .eyebrow { color: var(--accent); font-weight: 800; font-size: .78rem; letter-spacing: .16em; }
        .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .metric { background: rgba(15,23,42,.9); border: 1px solid #25314a; border-radius: 18px; padding: 16px; }
        .metric span { display: block; font-size: 1.6rem; font-weight: 800; color: #f8fafc; }
        .metric small { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
        .update-panel { display: grid; grid-template-columns: 0.8fr 1fr 1fr; gap: 12px; margin: 12px 0 12px 0; }
        .update-box { border: 1px solid #22304a; background: rgba(15,23,42,.86); border-radius: 18px; padding: 14px 16px; box-shadow: 0 10px 28px rgba(0,0,0,.22); }
        .update-main { border-color: rgba(56,189,248,.55); background: linear-gradient(135deg, rgba(14,116,144,.26), rgba(15,23,42,.9)); }
        .update-label { display: block; color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .11em; margin-bottom: 4px; }
        .update-box b { color: #f8fafc; font-size: 1.05rem; }
        .run-strip { border: 1px solid #22304a; background: rgba(8,13,25,.8); padding: 12px 16px; border-radius: 16px; color: #cbd5e1; margin-bottom: 18px; }
        .disclaimer-card { border: 1px solid rgba(249,115,115,.45); background: linear-gradient(135deg, rgba(127,29,29,.35), rgba(15,23,42,.95)); border-radius: 22px; padding: 24px; margin: 12px 0 22px 0; box-shadow: 0 18px 45px rgba(0,0,0,.28); }
        .disclaimer-card h2 { margin-top: 8px; }
        .badge-red { display:inline-block; background: rgba(248,113,113,.15); color:#fecaca; border:1px solid rgba(248,113,113,.45); border-radius: 999px; padding: 6px 11px; font-weight: 800; font-size: .75rem; letter-spacing: .1em; }
        .negative { color: #fecaca; }
        .muted { color: var(--muted); }
        .hero-caption { border: 1px solid #22304a; background: rgba(15,23,42,.75); padding: 10px 14px; border-radius: 14px; margin-bottom: 10px; color: #cbd5e1; }
        .zone-card { border: 1px solid var(--line); background: rgba(15,23,42,.86); border-radius: 22px; padding: 18px; margin-bottom: 16px; box-shadow: 0 12px 30px rgba(0,0,0,.22); }
        .zone-meta { color: var(--muted); border-bottom: 1px solid #1f2937; padding-bottom: 10px; margin-bottom: 14px; }
        .region-context { border: 1px solid #22304a; background: rgba(8,13,25,.78); border-radius: 14px; padding: 10px 12px; color: #cbd5e1; margin: 10px 0 14px 0; font-size: .94rem; line-height: 1.45; }
        .region-context b { color: #e0f2fe; }
        .graph-mini-title { min-height: 48px; font-size: .84rem; font-weight: 800; color: #e5e7eb; line-height: 1.15; margin-bottom: 6px; }
        div.stButton > button { width: 100%; border-radius: 12px; border: 1px solid #334155; background: #0f172a; color: #e5e7eb; font-weight: 700; }
        div.stButton > button:hover { border-color: #38bdf8; color: #e0f2fe; }
        [data-testid="stMetric"] { background: rgba(2,6,23,.35); border: 1px solid #22304a; border-radius: 16px; padding: 10px; }
        img { border-radius: 16px; }
        [data-testid="stRadio"] label span { color: #e5e7eb !important; }
        [data-testid="stSelectbox"] label { color: #cbd5e1 !important; font-weight: 700; }
        [data-baseweb="select"] > div { background-color: #f8fafc !important; color: #0f172a !important; }
        [data-baseweb="select"] span { color: #0f172a !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="TAMC–FRANJAMAR Monitor", layout="wide", initial_sidebar_state="expanded")
inject_css()

with st.sidebar:
    st.markdown("## Monitor settings")
    fixed_dir = get_resultados_dir()
    if fixed_dir is None:
        resultados_input = st.text_input(
            "Results folder",
            value="C:/Users/PC/Desktop/project/resultados",
            help="Main folder containing one subfolder per monitored zone.",
        )
        resultados_dir = Path(resultados_input).expanduser()
    else:
        resultados_dir = fixed_dir
        st.success(f"Fixed path: {resultados_dir}")

    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh interval", 10, 300, 60)
    show_all = st.toggle("Show all detected images", value=False)
    if st.button("Rescan files / clear cache"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Research use only · Non-predictive · Not for safety decisions")

if not resultados_dir.exists():
    st.error("The selected results folder does not exist.")
    st.stop()

zone_dirs = sorted(list_zone_dirs(resultados_dir), key=zone_sort_key, reverse=True)
if not zone_dirs:
    st.warning("No zone subfolders were found inside the results folder.")
    st.stop()

render_header(zone_dirs)
render_disclaimer()

main_tabs = st.tabs(["Monitor", "What is this?", "How it works", "DOI / Papers", "Disclaimer"])

with main_tabs[0]:
    render_hero_figure(zone_dirs)

    st.success(
        "Strictly fixed-parameter system: all regions are processed with identical configuration. "
        "Observed differences reflect system behaviour, not tuning."
    )

    st.info(
        "Global experimental monitoring layer: every region is processed with the same fixed pipeline, "
        "the same 24 h window and the same T−3 h delay. No region-specific tuning is applied."
    )

    layer = st.radio("Monitoring layer", LAYER_OPTIONS, horizontal=True, key="monitoring_layer")
    filtered = [z for z in zone_dirs if zone_matches_layer(z, layer)]

    if not filtered:
        st.warning(f"No regions detected for: {layer}")
    else:
        st.markdown("### Open region detail")
        st.caption(
            "Choose any monitored region in the selected layer. "
            "The featured Reykjanes figure above stays fixed as the scientific example, "
            "but the selector below controls the detailed region view."
        )

        selected_name = st.selectbox(
            "Region",
            [z.name for z in filtered],
            format_func=prettify_zone_name,
            key=f"zone_selector_{layer}",
        )
        selected_zone = next(z for z in filtered if z.name == selected_name)

        render_zone_detail(selected_zone, show_all=show_all)

        st.markdown("---")
        with st.expander(f"Show all regions in this layer ({len(filtered)})", expanded=False):
            for z in filtered:
                render_zone_card(z)

with main_tabs[1]:
    st.markdown("## What is this?")

    st.write(
        "This is an experimental scientific monitoring dashboard based on the TAMC–FRANJAMAR framework. "
        "TAMC–FRANJAMAR is a reproducible, retrospective statistical framework designed to analyze "
        "collective temporal organization in multistation systems."
    )

    st.write(
        "Instead of focusing on signal amplitude or conventional event detection, "
        "the framework studies how stations behave together in time, through temporal coherence, "
        "multistation synchrony, and collective statistical structure."
    )

    st.write(
        "All regions are processed using the SAME fixed pipeline, with identical parameters, "
        "applied consistently to earthquake monitoring regions, volcanic systems, fault/subduction zones, "
        "control windows, and monitoring runs. This ensures full reproducibility and avoids any region-specific tuning."
    )

    st.write(
        "The framework can be applied to real historical earthquakes (catalog ~200 events), "
        "volcanic systems, retrospective analyses, and near-real-time monitoring using a T−3 h delay."
    )

    st.caption("Monitoring mode is descriptive only and not used for statistical inference.")

    st.info("Focus: collective behaviour across stations, not individual amplitudes alone.")

    st.markdown("---")
    st.markdown("### What this is NOT")

    st.write(
        "This system is NOT an earthquake prediction tool, not an early warning system, "
        "and not a real-time operational monitoring platform."
    )

    st.write(
        "It does not estimate when an earthquake will occur, where it will occur, or how large it will be."
    )

    st.write(
        "It is not an amplitude-based detector and does not attempt to identify deterministic precursors."
    )

    st.write(
        "The framework is explicitly non-predictive and retrospective. "
        "Any predictive interpretation lies outside the scope of the method."
    )

with main_tabs[2]:
    st.markdown("## How it works")
    st.markdown(
        """
        The system processes **all selected regions using the same fixed pipeline**, 
        with identical parameters applied across all datasets.

        The workflow is **event-centered** and based on a common temporal reference (t = 0), 
        allowing direct comparison between different regions, events, and systems.

        Unlike conventional approaches that focus on signal amplitude or event detection, 
        this framework analyzes the emergence of **temporal coherence, multistation synchrony, 
        and collective statistical structure across the network**.

        The processing pipeline includes:

        - robust per-station normalization (median / MAD)
        - aggregation of standardized responses across stations
        - detection of multistation exceedances and collective activation
        - quantification of synchrony and temporal organization
        - comparison against control windows and null-model expectations

        All steps are applied identically to:

        - real historical earthquakes
        - volcanic episodes
        - matched control periods
        - monitoring windows

        This strict consistency ensures that observed differences reflect **true variations 
        in collective system behaviour**, not parameter tuning or methodological bias.

        The framework can be used in two modes:

        - **retrospective analysis** (event-centered, statistically evaluated)
        - **monitoring mode** using a near-real-time window with a **T−3 h delay**

        The T−3 h delay allows recent waveform data to consolidate across stations before processing, 
        ensuring stable and consistent inputs to the pipeline.

        Monitoring outputs are **descriptive only** and are not used for statistical inference.
        """
    )


    st.warning(
        "The five diagnostic panels should not be interpreted independently. "
        "Each graph captures only one aspect of the multistation response. "
        "The diagnostic value comes from the joint multistation pattern, not from any single panel in isolation."
    )

    st.markdown(
        """
### How to read the five panels

- **Extreme anomaly distribution**: highlights when the strongest standardized departures from baseline occur.
- **Multistation synchrony**: shows when several stations become active at the same time.
- **Mean anomaly and susceptibility**: summarizes the global evolution of the network response.
- **Station-resolved z-scores**: shows whether the activity is distributed across stations or dominated by isolated sensors.
- **Anomaly vs. synthetic tidal forcing**: provides a reference comparison with smooth external modulation; it is not a detector.

These panels are complementary. A single graph is not enough to claim structure.
        """
    )

with main_tabs[3]:
    st.markdown("## DOI / Papers")

    st.markdown(
        f"""
This dashboard is not an isolated demonstration. It is built around a set of reproducible TAMC–FRANJAMAR research records.
Each record includes the paper/materials plus downloadable Python code, configuration files, and instructions for local execution.

All monitoring outputs shown here are direct applications of the same fixed-parameter multistation pipeline:
**no region-specific tuning, same 24 h window, same T−3 h monitoring delay**.

---

### Seismic framework and monitoring implementation  
[{SEISMIC_DOI}]({SEISMIC_DOI})

This is the main reproducible implementation used for earthquake-style and regional monitoring runs.
It contains the Python pipeline, configuration files, station-selection logic, and the monitoring mode used to process recent 24 h windows.

It supports:

- analysis of real historical earthquakes
- a catalog-style workflow with **more than 200 earthquake events**
- matched control windows and null-model comparisons
- near-real-time monitoring using a **T−3 h delay**
- local execution from the downloadable Python code

This is the operational backbone for the seismic and regional monitoring layer of the dashboard.

---

### Volcanic framework and eruption analyses  
[{VOLCANIC_DOI}]({VOLCANIC_DOI})

This work applies the same TAMC–FRANJAMAR logic to volcanic systems, using fixed parameters and the same multistation statistical approach.
It analyzes real volcanic episodes and compares them under identical processing assumptions.

Examples of volcanic systems discussed or compatible with this monitoring layer include:

- Etna
- Stromboli
- Kīlauea
- Reykjanes / Icelandic systems
- La Palma-style volcanic monitoring contexts

The record includes the associated paper/materials and Python code so the analyses can be reproduced locally.
The same framework can also be used in monitoring mode, where recent waveform windows are processed descriptively without prediction.

---

### Multistation coupling regimes / extreme-event structure  
[{EXTREME_EVENT_DOI}]({EXTREME_EVENT_DOI})

This work develops the conceptual and comparative side of the framework: events are not characterized only by their external labels
such as earthquake, explosion, collapse, volcanic unrest, or anthropogenic source, but by their **collective multistation coupling structure**.

It includes examples of extreme or heterogeneous events and shows how they can separate into different dynamical regimes, such as:

- compact, strongly coupled multistation responses
- fragmented or weakly coupled responses
- temporally sustained network-coherent structures

This record also includes reproducible code and materials, allowing the coupling-regime analysis to be run locally.
It supports the interpretation of the monitoring dashboard as a system for comparing **structure over amplitude**.

---

### Reproducibility statement

All linked records provide reproducible material: paper text or documentation, Python code, configuration, and enough methodological detail to rerun the analyses locally.
The dashboard does not replace the papers; it is an experimental interface for visualizing the same family of fixed-pipeline multistation outputs.
"""
    )
with main_tabs[4]:
    render_disclaimer()

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
