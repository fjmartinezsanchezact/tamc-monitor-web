from __future__ import annotations

import html
import json
import random
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


def hide_streamlit_chrome() -> None:
    """Hide Streamlit footer/status UI without touching the app layout."""
    st.markdown("""
<style>
/* Classic footer and main menu */
footer {display: none !important; visibility: hidden !important; height: 0 !important;}
#MainMenu {display: none !important; visibility: hidden !important;}

/* Streamlit status / bottom widgets */
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stBottomBlockContainer"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="manage-app-button"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* Hosted / viewer badge variants */
.viewerBadge_container__1QSob,
.viewerBadge_link__1S137,
.viewerBadge_container__r5tak,
.viewerBadge_link__qRIco,
a[href*="streamlit.io"],
a[href*="share.streamlit.io"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Avoid ghost bottom gap + remove Streamlit top gap */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    padding-bottom: 1rem !important;
}

header,
[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

html, body {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_RESULTADOS_DIR = "web_data/latest"
EVENT_ANALYSIS_DIR = "event_analysis"
APP_HOME_URL = "https://franjamar-monitor.streamlit.app/"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
RAW_DATA_DIR_NAMES = {"raw", "mseed", "waveforms", "data"}

SEISMIC_DOI = "https://doi.org/10.5281/zenodo.19665949"
VOLCANIC_DOI = "https://doi.org/10.5281/zenodo.18525626"
EXTREME_EVENT_DOI = "https://doi.org/10.5281/zenodo.18649274"
FEEDBACK_EMAIL = "fjmartinezsanchezact@gmail.com"
FEEDBACK_SUBJECT = "TAMC-FRANJAMAR Monitor feedback"
APP_ICON_PATH = Path("icon_franjamar.png")
APP_VERSION_UTC = "2026-04-30 10:00 UTC"
APP_VERSION_ID = "v2026.04.30-1000UTC"
BUYMEACOFFEE_URL = "https://buymeacoffee.com/franjamar"

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
    "nerpio": "Nerpio (Albacete, Spain)",
    "minglanilla": "Minglanilla (Spain)",
    "lorca": "Lorca (Murcia, Spain)",
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
    "alaska_aleutian": "Alaska–Aleutian Subduction Zone, USA",
    "alaska": "Alaska–Aleutian Subduction Zone, USA",
    "aleutian": "Alaska–Aleutian Subduction Zone, USA",
    "kamchatka": "Kamchatka–Kuril Subduction Zone, Russia–Japan",
    "kuril": "Kamchatka–Kuril Subduction Zone, Russia–Japan",
    "sumatra": "Sumatra–Andaman Subduction Zone, Indonesia",
    "andaman": "Sumatra–Andaman Subduction Zone, Indonesia",
    "peru_ecuador": "Peru–Ecuador Subduction Zone",
    "ecuador": "Peru–Ecuador Subduction Zone",
    "peru": "Peru–Ecuador Subduction Zone",
    "anatolian": "North Anatolian Fault, Turkey",
    "north_anatolian": "North Anatolian Fault, Turkey",
}

ZONE_TIMEZONES = {
    "torremolinos": "Europe/Madrid",
    "nerpio": "Europe/Madrid",
    "minglanilla": "Europe/Madrid",
    "lorca": "Europe/Madrid",
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
    "alaska_aleutian": "America/Anchorage",
    "alaska": "America/Anchorage",
    "aleutian": "America/Adak",
    "kamchatka": "Asia/Kamchatka",
    "kuril": "Asia/Kamchatka",
    "sumatra": "Asia/Jakarta",
    "andaman": "Asia/Jakarta",
    "peru_ecuador": "America/Lima",
    "ecuador": "America/Guayaquil",
    "peru": "America/Lima",
    "anatolian": "Europe/Istanbul",
    "north_anatolian": "Europe/Istanbul",
}

VOLCANIC_KEYWORDS = [
    "volcan", "volcano", "volcanic", "eruption", "erupcion", "erupción",
    "reykjanes", "iceland", "la_palma", "lapalma", "yellowstone", "etna", "stromboli",
    "kilauea", "hawaii", "fuego", "popocatepetl", "popocat"
]

EARTHQUAKE_KEYWORDS = [
    "earthquake", "sismo", "terremoto", "miyako", "maule", "lorca", "torremolinos", "alboran", "nerpio", "minglanilla", "japan", "chile"
]

FAULT_SUBDUCTION_KEYWORDS = [
    "fault", "subduction", "san_andreas", "sanandreas", "cascadia", "hikurangi", "aegean", "greece", "turkey",
    "alaska", "aleutian", "kamchatka", "kuril", "sumatra", "andaman", "peru", "ecuador", "anatolian", "north_anatolian"
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
    "lorca": (
        "Why this region matters",
        "Spanish regional seismic system associated with the Alhama de Murcia fault context. Useful as a moderate-activity reference, between baseline regions and major global seismic systems."
    ),
    "torremolinos": (
        "Why this region matters",
        "Local reference region used for continuous monitoring under identical conditions. Acts as a baseline for comparison with other regions."
    ),
    "nerpio": (
        "Low-activity inland reference",
        "Inland low-activity region used as a real-world reference baseline under the same fixed processing conditions. Useful for comparison with active seismic, volcanic and fault/subduction systems."
    ),
    "minglanilla": (
        "Low-activity inland baseline",
        "Inland low-activity region used as a baseline reference under identical pipeline conditions. It helps contrast active systems with quieter monitoring regions without changing the method."
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


def get_event_analysis_dir() -> Path | None:
    """Fixed folder for retrospective real-event case studies.

    It is intentionally outside web_data/latest so daily monitor updates
    cannot overwrite these analyses.
    """
    p = Path(EVENT_ANALYSIS_DIR).expanduser().resolve()
    return p if p.exists() and p.is_dir() else None


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




def format_elapsed_since(dt_utc: datetime | None) -> str:
    """Human-readable elapsed time since a UTC datetime."""
    if dt_utc is None:
        return "not detected"
    now = datetime.now(timezone.utc)
    delta = now - dt_utc
    total_minutes = max(0, int(delta.total_seconds() // 60))
    if total_minutes < 1:
        return "just now"
    if total_minutes < 60:
        return f"{total_minutes} min ago"
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours < 48:
        return f"{hours} h {minutes} min ago" if minutes else f"{hours} h ago"
    days = hours // 24
    rem_hours = hours % 24
    return f"{days} d {rem_hours} h ago" if rem_hours else f"{days} d ago"

def zone_sort_key(zone_dir: Path) -> str:
    dt = extract_datetime_from_name(zone_dir.name)
    if dt is None:
        return "00000000000000"
    return dt.strftime("%Y%m%d%H%M%S")


def zone_intro_text(raw_name: str) -> str:
    name = raw_name.lower()
    if "nerpio" in name:
        return "Low-activity inland reference region used as a baseline under identical fixed-pipeline monitoring conditions."
    if "minglanilla" in name:
        return "Low-activity inland baseline region used for comparison with active seismic and volcanic systems."
    if "lorca" in name:
        return "Spanish moderate seismic reference region processed under the same fixed-pipeline monitoring conditions."
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


def event_location_from_name(raw_name: str) -> Dict[str, object] | None:
    """Approximate public event/reference coordinates for fixed case studies.

    These coordinates are used only for the visual station/event geometry map.
    They do not affect any TAMC calculation.
    """
    name = raw_name.lower().replace("-", "_")
    known = [
        (["tohoku", "20110311"], "Event location", 38.322, 142.369),
        (["maule", "20100227"], "Event location", -35.846, -72.719),
        (["miyako", "20260420"], "Event/reference location", 39.85, 143.22),
        (["kamchatka", "20250729"], "Event/reference location", 52.51, 160.32),
        (["lorca", "20110511"], "Event/reference location", 37.70, -1.67),
    ]
    for keys, label, lat, lon in known:
        if any(k in name for k in keys):
            return {"label": label, "lat": lat, "lon": lon}
    return None


def event_context_from_name(raw_name: str) -> Tuple[str, str]:
    """Short context for retrospective event-analysis cases."""
    name = raw_name.lower().replace("-", "_")
    if "tohoku" in name:
        return (
            "2011 Great Tohoku earthquake, Japan",
            "Major megathrust event associated with the Japan Trench. In this section it is used as a high-magnitude reference case processed with the same fixed pipeline as the other events.",
        )
    if "maule" in name:
        return (
            "2010 Maule earthquake, Chile",
            "Large subduction-zone earthquake used as a contrasting megathrust case. The goal is to compare multistation network structure, not magnitude.",
        )
    if "miyako" in name:
        return (
            "Miyako-region earthquake, Japan",
            "Modern Japanese event used to compare compact event-centered network responses against larger historical earthquakes under identical processing.",
        )
    if "kamchatka" in name:
        return (
            "Kamchatka Peninsula earthquake",
            "Large subduction-zone case in the Kamchatka–Kuril system, useful for comparing clustered extremes, synchrony and post-event structure.",
        )
    return (
        "Real earthquake event analysis",
        "Retrospective case study processed with the same fixed TAMC–FRANJAMAR pipeline. This is descriptive analysis, not warning, forecasting or risk estimation.",
    )


def render_event_context_block(raw_name: str) -> None:
    title, body = event_context_from_name(raw_name)
    st.markdown(
        f"""
<div class="event-context-card">
  <div class="event-context-kicker">REAL EVENT CASE STUDY</div>
  <div class="event-context-title">{html.escape(title)}</div>
  <div class="event-context-body">{html.escape(body)}</div>
  <div class="event-context-note">Same fixed pipeline · no event-specific tuning · score describes structure, not magnitude.</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _map_zoom_from_span(lat_values: List[float], lon_values: List[float]) -> int:
    span = max((max(lat_values) - min(lat_values)) if lat_values else 0, (max(lon_values) - min(lon_values)) if lon_values else 0)
    if span < 0.4:
        return 8
    if span < 1.0:
        return 7
    if span < 3.0:
        return 6
    if span < 8.0:
        return 5
    if span < 18.0:
        return 4
    return 3


def render_station_geometry_map(details: List[Dict[str, object]], raw_name: str, include_event: bool = False) -> None:
    """Render a reliable Leaflet map: stations in blue, event/reference in red.

    This avoids Streamlit's native st.map/pydeck rendering issues on some
    dark-theme/mobile deployments and keeps the map centered on the local
    event/station geometry instead of the whole globe.
    """
    rows: List[Dict[str, object]] = []
    distances: List[float] = []

    for d in details:
        lat = d.get("lat")
        lon = d.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            continue

        dist = d.get("dist_km")
        try:
            if dist is not None:
                distances.append(float(dist))
        except Exception:
            pass

        rows.append({
            "lat": lat_f,
            "lon": lon_f,
            "label": str(d.get("code", "station")),
            "kind": "station",
        })

    event_point = event_location_from_name(raw_name) if include_event else None
    if event_point is not None:
        try:
            rows.append({
                "lat": float(event_point["lat"]),
                "lon": float(event_point["lon"]),
                "label": str(event_point.get("label", "Event/reference")),
                "kind": "event",
            })
        except Exception:
            event_point = None

    if not rows:
        st.info("No station coordinates available for the geometry map.")
        return

    station_count = len([r for r in rows if r.get("kind") == "station"])
    title = "### 🗺️ Station/event geometry map" if event_point else "### 🗺️ Station geometry map"
    st.markdown(title)
    if event_point:
        st.caption("Blue points are stations. The larger red point is the event/reference location. The map is centered on this local geometry.")
    else:
        st.caption("Blue points are stations from `fixed_stations.json`. The map is centered on the local station geometry.")

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    zoom = _map_zoom_from_span(lats, lons)
    markers_json = json.dumps(rows, ensure_ascii=False)
    map_id = "tamc_map_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)[:80] + "_" + str(abs(hash(raw_name)) % 100000)

    map_html = f"""
<div id="{map_id}" style="height:430px; width:100%; border-radius:18px; overflow:hidden; border:1px solid rgba(96,165,250,0.35); background:#dbeafe;"></div>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function() {{
  const markers = {markers_json};
  const map = L.map('{map_id}', {{ scrollWheelZoom: false }}).setView([{center_lat:.6f}, {center_lon:.6f}], {zoom});
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  const bounds = [];
  markers.forEach(function(m) {{
    const isEvent = m.kind === 'event';
    const marker = L.circleMarker([m.lat, m.lon], {{
      radius: isEvent ? 11 : 6,
      color: isEvent ? '#991b1b' : '#1d4ed8',
      weight: isEvent ? 3 : 2,
      fillColor: isEvent ? '#ef4444' : '#2563eb',
      fillOpacity: isEvent ? 0.95 : 0.85
    }}).addTo(map);
    marker.bindTooltip(m.label, {{permanent:false, direction:'top'}});
    marker.bindPopup((isEvent ? '<b>Event/reference</b><br>' : '<b>Station</b><br>') + m.label);
    bounds.push([m.lat, m.lon]);
  }});

  if (bounds.length > 1) {{
    map.fitBounds(bounds, {{ padding: [35, 35], maxZoom: 8 }});
  }}

  setTimeout(function() {{ map.invalidateSize(); }}, 250);
}})();
</script>
"""

    map_col, info_col = st.columns([0.70, 0.30])
    with map_col:
        components.html(map_html, height=450, scrolling=False)
        hide_streamlit_chrome()

    with info_col:
        st.markdown(
            f"""
<div class="station-map-card">
  <div class="station-map-kicker">NETWORK GEOMETRY</div>
  <div class="station-map-value">{station_count}</div>
  <div class="station-map-label">stations mapped</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if event_point:
            st.metric("Event/reference", "red point")
        if distances:
            st.metric("Distance range", f"{min(distances):.1f}–{max(distances):.1f} km")
            st.metric("Mean distance", f"{sum(distances)/len(distances):.1f} km")
        st.caption("Geometry is read automatically from the run metadata; the map is not manually drawn.")


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


def render_station_metadata_visible(details: List[Dict[str, object]], source_label: str, source_files: List[str]) -> None:
    """Render JSON-derived station metadata directly under the graphs.

    This is intentionally visible on the main page because the station set is part
    of the scientific result: it documents which network geometry produced the
    five panels above.
    """
    if not details:
        st.warning("No station metadata JSON was detected for this region.")
        return

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

    st.markdown("### JSON-derived station metadata")
    st.caption(
        f"Source: `{source_label}` · shown directly because the station set and geometry are part of the reproducible output."
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Coordinates and distances correspond to the fixed station set used for this run. "
        "All stations are selected automatically by the pipeline under identical criteria."
    )

    st.markdown("**Station codes used in this run**")
    st.code("  ".join(str(d.get("code", "")) for d in details), language="text")

    with st.expander("Show JSON/source file name(s)", expanded=False):
        for f in source_files:
            st.write(Path(f).name)


def pick_hero(zone_dirs: List[Path]) -> Tuple[Path | None, str | None, str | None]:
    """Pick a dynamic hero figure.

    50% of the time it prioritizes high-impact public-interest regions
    (San Andreas Fault or Yellowstone). The other 50% is randomly selected
    from the remaining available monitoring outputs. This keeps the front
    page alive while avoiding a fixed hand-picked example.
    """
    if not zone_dirs:
        return None, None, None

    priority_zones = [
        z for z in zone_dirs
        if any(k in z.name.lower() for k in ["san_andreas", "sanandreas", "yellowstone"])
    ]
    other_zones = [z for z in zone_dirs if z not in priority_zones]

    # 50% San Andreas / Yellowstone, 50% the rest.
    if priority_zones and random.random() < 0.5:
        candidate_zones = priority_zones[:]
    else:
        candidate_zones = other_zones[:] if other_zones else zone_dirs[:]

    random.shuffle(candidate_zones)

    preferred_slots = [
        "Extreme anomaly distribution",
        "Multistation synchrony",
        "Mean anomaly and susceptibility",
        "Station-resolved z-scores",
        "Anomaly vs. synthetic tidal forcing",
    ]

    for z in candidate_zones:
        assigned, _ = load_zone_images(str(z.resolve()))
        for slot in preferred_slots:
            img = assigned.get(slot)
            if img and Path(img).exists():
                return z, slot, img

    # Last fallback: scan every available region.
    for z in zone_dirs:
        assigned, _ = load_zone_images(str(z.resolve()))
        for slot in preferred_slots:
            img = assigned.get(slot)
            if img and Path(img).exists():
                return z, slot, img

    return None, None, None


def render_disclaimer() -> None:
    """Render a stable native Streamlit research/disclaimer block.

    This intentionally avoids raw HTML in the body so Streamlit Cloud cannot
    display tags as text/code.
    """
    st.markdown("### 🔬 EXPERIMENTAL · NON-PREDICTIVE · BETA VERSION")
    st.markdown(
        f"""

**Real updated data · fixed reproducible pipeline · no region-specific tuning**

It analyzes **collective behavior** and **multistation coherence** in seismic networks using rolling **24 h windows** with a short consolidation delay (**T−1 h**).

**Key observation:** the dashboard highlights patterns that emerge at **network level**, not only at isolated single-station level.

**Methodology:** identical fixed pipeline · reproducible outputs · direct comparability across regions.

⚠️ **This system does not forecast earthquakes, eruptions, timing, magnitude or risk.**

Its purpose is to characterize **statistical structure**, detect **deviations from baseline behavior**, and identify emerging **network regimes** for exploratory research.
        """
    )

def select_preferred_region(zone_dirs: List[Path], keywords: List[str]) -> None:
    """Select the newest available region matching any keyword."""
    for z in zone_dirs:
        name = z.name.lower()
        if any(k.lower() in name for k in keywords):
            st.session_state.selected_layer = "All regions"
            st.session_state.selected_zone_name = z.name
            return




def _summary_for_zone(zone_dir: Path) -> Dict[str, object]:
    """Compact app-facing summary from network_summary.json plus region metadata."""
    summary = load_network_summary(str(zone_dir.resolve())) or {}
    interp = summary.get("joint_interpretation") or summary.get("interpretation") or {}
    state = str(interp.get("network_state", "Not available"))
    score_raw = interp.get("descriptive_score_0_100")
    try:
        score = float(score_raw) if score_raw is not None else None
    except Exception:
        score = None
    color, short_state = get_network_state_style(state)
    score_level = str(interp.get("descriptive_score_level", "not available")).upper()
    return {
        "zone": zone_dir,
        "name": compact_region_name_for_button(zone_dir.name),
        "display": clean_zone_display_name(zone_dir.name),
        "state": state,
        "short_state": short_state,
        "score": score,
        "score_level": score_level,
        "color": color,
        "rtype": physical_region_type(zone_dir.name),
        "flag": get_flag(get_country_code_from_name(zone_dir.name)),
    }


def _sorted_region_summaries(zone_dirs: List[Path]) -> List[Dict[str, object]]:
    items = [_summary_for_zone(z) for z in zone_dirs]
    return sorted(
        items,
        key=lambda x: (9999 if x.get("score") is None else -float(x["score"]), str(x.get("name", ""))),
    )


def _global_status_label(items: List[Dict[str, object]]) -> Tuple[str, str, str]:
    scores = [float(x["score"]) for x in items if x.get("score") is not None]
    if not scores:
        return "⚪ No summary", "Network summaries not available yet", "#94a3b8"
    top = max(scores)
    if top >= 75:
        return "🔴 High network organization", "At least one region shows a strong descriptive structure score", "#ef4444"
    if top >= 50:
        return "🟠 Moderate network organization", "Some regions show organized or transient multistation structure", "#f59e0b"
    if top >= 30:
        return "🟡 Low–moderate network organization", "Mostly background with some localized structure", "#eab308"
    return "🟢 Mostly quiet network state", "No region currently dominates the descriptive structure ranking", "#22c55e"


def open_region_from_summary(it: Dict[str, object]) -> None:
    st.session_state.page = "monitor"
    st.session_state.show_region_catalog = False
    st.session_state.selected_layer = "All regions"
    st.session_state.selected_zone_name = it["zone"].name
    request_scroll("focused_region")


def region_query_url(zone_name: str) -> str:
    """Build an internal URL that opens a region reliably after reload."""
    return f"?page=monitor&region={quote(str(zone_name), safe='')}#focused_region"


def apply_query_params_to_state(zone_dirs: List[Path]) -> None:
    """Read URL parameters and map them to Streamlit session state."""
    try:
        params = st.query_params
    except Exception:
        return

    page = params.get("page", None)
    if isinstance(page, list):
        page = page[0] if page else None
    if page in {"monitor", "events", "what", "how", "read", "roadmap", "feedback", "papers"}:
        st.session_state.page = page

    region = params.get("region", None)
    if isinstance(region, list):
        region = region[0] if region else None
    if region:
        match = next((z for z in zone_dirs if z.name == region), None)
        if match is not None:
            st.session_state.page = "monitor"
            st.session_state.show_region_catalog = False
            st.session_state.selected_layer = "All regions"
            st.session_state.selected_zone_name = match.name
            request_scroll("focused_region")

    catalog = params.get("catalog", None)
    if isinstance(catalog, list):
        catalog = catalog[0] if catalog else None
    if catalog == "1":
        st.session_state.page = "monitor"
        st.session_state.show_region_catalog = True
        st.session_state.selected_layer = "All regions"
        request_scroll("region_selector")

    ranking = params.get("ranking", None)
    if isinstance(ranking, list):
        ranking = ranking[0] if ranking else None
    if ranking == "1":
        st.session_state.page = "monitor"
        st.session_state.show_full_ranking = True
        request_scroll("full_ranking")


def render_global_status_radar(zone_dirs: List[Path]) -> None:
    """App-facing radar plus native Streamlit navigation controls.

    The visual cards are clickable internal links. The Streamlit buttons below
    remain as reliable fallbacks.
    """
    items_sorted = _sorted_region_summaries(zone_dirs)
    top_items = items_sorted[:6]
    top_item = top_items[0] if top_items else None
    secondary_items = top_items[1:]
    label, subtitle, global_color = _global_status_label(items_sorted)

    def score_text(it: Dict[str, object]) -> str:
        score = it.get("score")
        return "—" if score is None else f"{float(score):.1f}"

    def href_for(it: Dict[str, object]) -> str:
        return region_query_url(str(it["zone"].name))

    top_html = ""
    if top_item:
        top_flag = f"{top_item.get('flag')} " if top_item.get("flag") else ""
        top_html = (
            f'<a class="radar-card-link" href="{href_for(top_item)}" target="_self" title="Open {html.escape(str(top_item["name"]))}">'
            f'<div class="radar-top-card clickable-radar-card" style="border-color:{top_item["color"]};">'
            f'<div class="radar-top-badge">TOP REGION · TAP TO OPEN</div>'
            f'<div class="radar-top-name">{html.escape(top_flag + str(top_item["name"]))}</div>'
            f'<div class="radar-top-meta">{html.escape(str(top_item["rtype"]))} · descriptive score</div>'
            f'<div class="radar-top-score"><span>{html.escape(score_text(top_item))}</span><small>/100</small></div>'
            f'<div class="radar-top-state" style="color:{top_item["color"]};">{html.escape(str(top_item["short_state"]))}</div>'
            f'</div></a>'
        )

    cards_html = ""
    for it in secondary_items:
        flag = f"{it.get('flag')} " if it.get("flag") else ""
        cards_html += (
            f'<a class="radar-card-link" href="{href_for(it)}" target="_self" title="Open {html.escape(str(it["name"]))}">'
            f'<div class="radar-region-card clickable-radar-card" style="border-color:{it["color"]};">'
            f'<div class="radar-region-name">{html.escape(flag + str(it["name"]))}</div>'
            f'<div class="radar-region-type">{html.escape(str(it["rtype"]))}</div>'
            f'<div class="radar-score-row"><span>{html.escape(score_text(it))}</span><small>/100</small></div>'
            f'<div class="radar-state" style="color:{it["color"]};">{html.escape(str(it["short_state"]))}</div>'
            f'</div></a>'
        )

    radar_html = f"""
<div class="global-radar-card">
  <div class="radar-head-row">
    <div>
      <div class="radar-kicker">GLOBAL STATUS RADAR</div>
      <div class="radar-main" style="color:{global_color};">{html.escape(label)}</div>
      <div class="radar-subtitle">{html.escape(subtitle)} · descriptive, non-predictive ranking</div>
      <div class="radar-tap-hint">Tap/click a radar card or use the buttons below to open region details.</div>
    </div>
  </div>
  <div class="radar-layout">{top_html}<div class="radar-grid">{cards_html}</div></div>
</div>
    """
    st.markdown(radar_html, unsafe_allow_html=True)

    # Classification control remains native because it was already working reliably.
    rank_left, rank_right = st.columns([0.70, 0.30])
    with rank_right:
        if st.button("📋 Classification: all regions", key="radar_full_ranking_top", use_container_width=True):
            st.session_state.page = "monitor"
            st.session_state.show_full_ranking = not st.session_state.get("show_full_ranking", False)
            st.session_state.show_region_catalog = False
            request_scroll("full_ranking")
            st.rerun()

    if st.session_state.get("show_full_ranking", False):
        render_region_classification_table(items_sorted)


def render_region_classification_table(items_sorted: List[Dict[str, object]]) -> None:
    """Full descending classification by descriptive score, with open buttons."""
    st.markdown("<div id='full_ranking'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Region classification · highest to lowest score")
    st.caption("Descriptive ranking only. It is not a warning, forecast or risk estimate.")

    for idx, it in enumerate(items_sorted, start=1):
        score = it.get("score")
        score_txt = "—" if score is None else f"{float(score):.1f}"
        flag = f"{it.get('flag')} " if it.get("flag") else ""
        cols = st.columns([0.08, 0.34, 0.16, 0.22, 0.20])
        cols[0].markdown(f"**#{idx}**")
        cols[1].markdown(f"**{flag}{html.escape(str(it['name']))}**")
        cols[2].markdown(f"`{score_txt}/100`")
        cols[3].markdown(html.escape(str(it.get("short_state", ""))))
        with cols[4]:
            if st.button("Open", key=f"ranking_open_{idx}_{it['zone'].name}", use_container_width=True):
                open_region_from_summary(it)
                st.rerun()

def render_quick_guide_wizard() -> None:
    """Small guided onboarding wizard shown from the Start Here block.

    It tells first-time users what to do, then points them to the detailed
    interpretation and the JSON-derived station metadata.
    """
    if "quick_guide_open" not in st.session_state:
        st.session_state.quick_guide_open = False
    if "quick_guide_step" not in st.session_state:
        st.session_state.quick_guide_step = 1

    st.markdown(
        """
<style>
.quick-guide-button-wrap {
    margin-top: 0.75rem;
    display: inline-flex;
}
.quick-guide-button-wrap a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    padding: 0.56rem 1.05rem;
    border-radius: 999px;
    border: 1px solid rgba(244,114,182,0.92);
    background: linear-gradient(135deg, rgba(236,72,153,0.24), rgba(59,130,246,0.16));
    color: #fbcfe8 !important;
    text-decoration: none !important;
    font-weight: 900;
    letter-spacing: 0.04em;
    box-shadow: 0 0 0 1px rgba(244,114,182,0.18), 0 0 22px rgba(236,72,153,0.24);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    animation: quickGuidePulse 2.2s infinite;
}
.quick-guide-button-wrap a:hover {
    transform: translateY(-1px);
    border-color: rgba(251,207,232,1);
    box-shadow: 0 0 0 1px rgba(244,114,182,0.38), 0 0 34px rgba(236,72,153,0.42);
    color: #ffffff !important;
}
.quick-guide-card {
    margin: 1.0rem 0 0.8rem 0;
    padding: 1.15rem 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(244,114,182,0.72);
    background: linear-gradient(135deg, rgba(83,13,55,0.34), rgba(15,23,42,0.96));
    box-shadow: 0 0 0 1px rgba(244,114,182,0.14), 0 16px 34px rgba(0,0,0,0.28), 0 0 28px rgba(236,72,153,0.18);
    animation: quickGuideCardGlow 1.35s ease-out 1;
}
.quick-guide-top-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:0.75rem;
    margin-bottom:0.8rem;
}
.quick-guide-kicker {
    color:#f472b6;
    font-size:0.78rem;
    font-weight:900;
    letter-spacing:0.15em;
}
.quick-guide-progress {
    color:#cbd5e1;
    font-size:0.78rem;
    font-weight:800;
    border:1px solid rgba(244,114,182,0.34);
    border-radius:999px;
    padding:0.22rem 0.62rem;
    background:rgba(15,23,42,0.72);
}
.quick-guide-title {
    color:#ffffff;
    font-size:1.08rem;
    font-weight:900;
    margin-bottom:0.42rem;
}
.quick-guide-body {
    color:#e5e7eb;
    line-height:1.58;
    font-size:0.98rem;
    max-width:980px;
}
.quick-guide-body b { color:#93c5fd; }
.quick-guide-note {
    margin-top:0.75rem;
    color:#fcd34d;
    font-size:0.86rem;
    font-weight:800;
}
.quick-guide-next-target {
    margin-top:0.75rem;
    padding:0.72rem 0.9rem;
    border-radius:14px;
    border:1px solid rgba(56,189,248,0.32);
    background:rgba(8,47,73,0.30);
    color:#bae6fd;
    font-size:0.9rem;
    line-height:1.45;
}
@keyframes quickGuidePulse {
    0% { box-shadow: 0 0 0 0 rgba(236,72,153,0.42), 0 0 22px rgba(236,72,153,0.24); }
    70% { box-shadow: 0 0 0 9px rgba(236,72,153,0), 0 0 28px rgba(236,72,153,0.30); }
    100% { box-shadow: 0 0 0 0 rgba(236,72,153,0), 0 0 22px rgba(236,72,153,0.24); }
}
@keyframes quickGuideCardGlow {
    0% { transform: translateY(4px); box-shadow: 0 0 0 1px rgba(244,114,182,0.26), 0 0 44px rgba(236,72,153,0.42); }
    100% { transform: translateY(0); box-shadow: 0 0 0 1px rgba(244,114,182,0.14), 0 16px 34px rgba(0,0,0,0.28), 0 0 28px rgba(236,72,153,0.18); }
}
@media (max-width: 900px) {
    .quick-guide-button-wrap { width:100%; }
    .quick-guide-button-wrap a { width:100%; }
    .quick-guide-card { padding:1rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # Stable anchor used to keep the viewport near the guide after Streamlit reruns.
    st.markdown("<div id='quick_guide_anchor'></div>", unsafe_allow_html=True)

    # Styled HTML link for a reliable pink framed button. The query parameter is
    # consumed below and avoids fragile custom JavaScript callbacks.
    st.markdown(
        """
<div class="quick-guide-button-wrap">
  <a href="?quickguide=1#quick_guide_anchor" target="_self">🧭 Quick guide</a>
</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        quickguide_param = st.query_params.get("quickguide", None)
    except Exception:
        quickguide_param = None

    if quickguide_param == "1" and not st.session_state.quick_guide_open:
        st.session_state.quick_guide_open = True
        st.session_state.quick_guide_step = 1

    if not st.session_state.quick_guide_open:
        return

    # Streamlit reruns can jump to the top of the page after opening/advancing the guide.
    # This small script recenters the viewport on the guide block.
    components.html(
        """
<script>
setTimeout(() => {
  const el = window.parent.document.getElementById("quick_guide_anchor");
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}, 180);
</script>
        """,
        height=0,
    )

    steps = {
        1: {
            "title": "1️⃣ Choose a monitoring region",
            "body": (
                "Start with <b>San Andreas</b>, <b>Yellowstone</b>, or open <b>Explore all regions</b>. "
                "This loads the latest fixed-pipeline output for one network, so every region can be compared under the same rules."
            ),
            "target": "Tip: use the quick region buttons first, then compare with other regions later."
        },
        2: {
            "title": "2️⃣ Inspect the five core outputs",
            "body": (
                "Look at synchrony, extreme anomalies, mean anomaly/susceptibility, station z-scores, and the forcing comparison. "
                "You are not reading one isolated graph: you are checking how the station network behaves as a whole."
            ),
            "target": "A useful signal should make sense across several panels, not only in one plot."
        },
        3: {
            "title": "3️⃣ Look for consistency across panels",
            "body": (
                "Check whether several signals change together: higher synchrony, clustered extremes, and simultaneous station deviations. "
                "Short spikes can be interesting, but sustained or repeated agreement across outputs is more informative."
            ),
            "target": "The dashboard is designed to describe collective structure, not single-station noise."
        },
        4: {
            "title": "4️⃣ Read and verify the interpretation",
            "body": (
                "Read the <b>network state</b> and <b>structure score</b> as the summary. Then open <b>Deeper network analysis</b> "
                "for the detailed explanation. Finally, review <b>JSON-derived station metadata</b> to see which stations, coordinates, "
                "and distances produced the result."
            ),
            "target": "Next: open “Deeper network analysis”, then review “JSON-derived station metadata”."
        },
    }

    step = max(1, min(4, int(st.session_state.quick_guide_step)))
    item = steps[step]

    st.markdown(
        f"""
<div class="quick-guide-card">
  <div class="quick-guide-top-row">
    <div class="quick-guide-kicker">QUICK GUIDE</div>
    <div class="quick-guide-progress">Step {step} / 4</div>
  </div>
  <div class="quick-guide-title">{item['title']}</div>
  <div class="quick-guide-body">{item['body']}</div>
  <div class="quick-guide-next-target">{item['target']}</div>
  <div class="quick-guide-note">Pattern recognition · not prediction · descriptive research only</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    c_back, c_next, c_close = st.columns([0.24, 0.46, 0.30])
    with c_back:
        if step > 1:
            if st.button("← Back", key="quick_guide_back", use_container_width=True):
                st.session_state.quick_guide_step = step - 1
                st.rerun()
    with c_next:
        if step < 4:
            if st.button("Next step →", key="quick_guide_next", use_container_width=True):
                st.session_state.quick_guide_step = step + 1
                st.rerun()
        else:
            if st.button("Finish guide ✅", key="quick_guide_finish", use_container_width=True):
                st.session_state.quick_guide_open = False
                st.session_state.quick_guide_step = 1
                st.rerun()
    with c_close:
        if st.button("Close", key="quick_guide_close", use_container_width=True):
            st.session_state.quick_guide_open = False
            st.session_state.quick_guide_step = 1
            st.rerun()


def render_start_here_block(zone_dirs: List[Path]) -> None:
    """Guided entry point shown before the catalogue."""
    st.markdown("<div id='start_here'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="start-here-card">
            <div class="button-section-title">START HERE</div>
            <div class="start-here-text">
                Start with a reference region. Inspect the five outputs, then read the network state classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_quick_guide_wizard()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⚡ San Andreas", key="quick_san_andreas", use_container_width=True):
            st.session_state.page = "monitor"
            st.session_state.show_region_catalog = False
            st.session_state.show_full_ranking = False
            select_preferred_region(zone_dirs, ["san_andreas", "sanandreas"])
            request_scroll("focused_region")
            st.rerun()
    with c2:
        if st.button("🌋 Yellowstone", key="quick_yellowstone", use_container_width=True):
            st.session_state.page = "monitor"
            st.session_state.show_region_catalog = False
            st.session_state.show_full_ranking = False
            select_preferred_region(zone_dirs, ["yellowstone"])
            request_scroll("focused_region")
            st.rerun()
    with c3:
        if st.button("🌍 Explore all regions", key="quick_explore_all_regions", use_container_width=True):
            st.session_state.page = "monitor"
            st.session_state.show_region_catalog = True
            st.session_state.show_full_ranking = False
            st.session_state.selected_layer = "All regions"
            request_scroll("region_selector")
            st.rerun()

def render_header(zone_dirs: List[Path]) -> None:
    """Render main hero, framework DOI, beta/version label and data-window cards."""
    latest_dt = extract_datetime_from_name(zone_dirs[0].name) if zone_dirs else None
    latest_txt = latest_dt.strftime("%Y-%m-%d %H:%M UTC") if latest_dt else "not detected"
    data_age_txt = format_elapsed_since(latest_dt)
    if latest_dt:
        window_start_txt = (latest_dt - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M UTC")
        window_txt = f"{window_start_txt} → {latest_txt}"
    else:
        window_txt = "not detected"

    volcanic_count = sum(1 for z in zone_dirs if physical_region_type(z.name) == "Volcanic")
    subduction_count = sum(1 for z in zone_dirs if physical_region_type(z.name) == "Subduction")
    fault_count = sum(1 for z in zone_dirs if physical_region_type(z.name) == "Fault")

    st.markdown(
        """
<style>
.two-update-boxes {
    grid-template-columns: minmax(260px, 0.82fr) minmax(360px, 1.18fr) !important;
}
.update-box small {
    display:block;
    margin-top:0.35rem;
    color:rgba(203,213,225,0.78);
    font-size:0.86rem;
    font-weight:600;
}
.framework-line {
    margin-top: 0.65rem;
    color: rgba(226,232,240,0.88);
    font-size: 0.98rem;
    line-height: 1.45;
}
.framework-line a {
    color:#60a5fa !important;
    font-weight:800;
    text-decoration:none;
    border-bottom:1px solid rgba(96,165,250,0.55);
}
.framework-line a:hover { color:#93c5fd !important; border-bottom-color:#93c5fd; }
.hero-status-row {
    display:flex;
    flex-wrap:wrap;
    gap:0.55rem;
    margin-top:0.85rem;
}
.hero-pill {
    display:inline-flex;
    align-items:center;
    gap:0.35rem;
    padding:0.34rem 0.72rem;
    border-radius:999px;
    border:1px solid rgba(56,189,248,0.45);
    background:rgba(8,47,73,0.34);
    color:#bae6fd;
    font-size:0.78rem;
    font-weight:900;
    letter-spacing:0.07em;
    text-transform:uppercase;
}
.hero-pill.beta {
    border-color:rgba(244,114,182,0.55);
    background:rgba(80,7,36,0.35);
    color:#fbcfe8;
}
.hero-pill.version {
    border-color:rgba(148,163,184,0.35);
    background:rgba(15,23,42,0.52);
    color:#cbd5e1;
    letter-spacing:0.04em;
}
.research-disclaimer-card {
    margin: 1.4rem 0 2.0rem 0;
    padding: 1.35rem 1.45rem;
    border-radius: 18px;
    border: 1px solid rgba(236,72,153,0.78);
    background: linear-gradient(135deg, rgba(92,18,38,0.72), rgba(8,16,35,0.94));
    box-shadow: 0 0 0 1px rgba(59,130,246,0.08), 0 14px 34px rgba(0,0,0,0.28);
}
.research-badge-row { display:flex; flex-wrap:wrap; gap:0.6rem; align-items:center; margin-bottom:1.05rem; }
.research-badge, .beta-badge {
    display:inline-block;
    padding: 0.42rem 0.95rem;
    border-radius: 999px;
    font-weight: 900;
    letter-spacing: 0.14em;
    font-size: 0.82rem;
}
.research-badge {
    border: 1px solid rgba(248,113,113,0.82);
    background: rgba(127,29,29,0.38);
    color: #fecdd3;
}
.beta-badge {
    border: 1px solid rgba(56,189,248,0.7);
    background: rgba(8,47,73,0.42);
    color:#bae6fd;
}
.research-lead {
    font-size:1.08rem;
    line-height:1.62;
    color:#e5e7eb;
    max-width: 1180px;
}
.research-lead.secondary { margin-top:0.45rem; }
.research-lead b { color:#60a5fa; }
.research-lead a { color:#93c5fd !important; font-weight:900; text-decoration:none; border-bottom:1px solid rgba(147,197,253,0.65); }
.scope-lines {
    margin-top:1.05rem;
    padding-top:1.05rem;
    border-top:1px solid rgba(148,163,184,0.25);
    display:grid;
    grid-template-columns:1fr;
    gap:0.74rem;
    color:#dbeafe;
    font-size:0.98rem;
    line-height:1.55;
}
.scope-line { display:flex; gap:0.7rem; align-items:flex-start; }
.scope-icon {
    width:1.9rem;
    height:1.9rem;
    min-width:1.9rem;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    border-radius:999px;
    background:rgba(14,165,233,0.14);
    border:1px solid rgba(56,189,248,0.35);
}
.risk-line {
    margin-top:1.05rem;
    padding-top:1rem;
    border-top:1px solid rgba(148,163,184,0.25);
    color:#fecaca;
    font-size:1.02rem;
    line-height:1.55;
}
@media (max-width: 900px) {
    .two-update-boxes { grid-template-columns: 1fr !important; }
    .research-disclaimer-card { padding:1rem; }
    .research-lead { font-size:1rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="top-hero">
            <div>
                <div class="eyebrow">NEAR-REAL-TIME MONITORING (T−1 DELAY)</div>
                <h1>GLOBAL SEISMIC MULTISTATION COHERENCE MONITOR</h1>
                <p class="plain-definition"><b>Network behaving as one.</b> Fixed-pipeline coherence monitoring across regions.</p>
                <div class="framework-line">
                    Framework: <b>TAMC–FRANJAMAR v3</b> ·
                    <a href="{SEISMIC_DOI}" target="_blank" rel="noopener noreferrer">DOI: 10.5281/zenodo.19665949</a>
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric"><span>{len(zone_dirs)}</span><small>regions monitored</small></div>
                <div class="metric"><span>{volcanic_count}</span><small>volcanic systems</small></div>
                <div class="metric"><span>{subduction_count}</span><small>subduction zones</small></div>
                <div class="metric"><span>{fault_count}</span><small>fault systems</small></div>
            </div>
        </div>

        <div class="update-panel two-update-boxes">
            <div class="update-box update-main">
                <span class="update-label">Time since latest data</span>
                <b>{data_age_txt}</b>
                <small>Rolling data-window age</small>
            </div>
            <div class="update-box">
                <span class="update-label">Latest data window</span>
                <b>{window_txt}</b>
                <small>24 h window · T−1 h consolidation delay</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_version_status_card() -> None:
    """Compact status/version strip shown outside the hero.

    It is intentionally rendered as plain status text, not as separate pill
    buttons, because these labels are informational and not clickable.
    """
    st.markdown(
        f"""
        <div class="version-status-card">
            <span class="version-status-beta">BETA VERSION</span>
            <span class="version-separator">·</span>
            <span class="version-status-data">Automatically updated daily · fixed pipeline</span>
            <span class="version-separator">·</span>
            <span class="version-status-version">{APP_VERSION_ID} · APP.PY UPLOADED {APP_VERSION_UTC}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_figure(zone_dirs: List[Path]) -> None:
    """Render a compact visible featured figure.

    Small card only: region name + plot. It should give a visual hook without
    occupying too much vertical space before the region selector.
    """
    hero_zone, hero_slot, hero_img = pick_hero(zone_dirs)
    if not hero_zone or not hero_img:
        return

    st.markdown(
        f"""
        <div class="compact-feature-card">
            <div class="compact-feature-title">Featured output</div>
            <div class="compact-feature-caption">
                <b>{prettify_zone_name(hero_zone.name)}</b> · {hero_slot}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1.35, 1.3, 1.35])
    with mid:
        st.image(hero_img, width=460)


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
    c3.metric("Mode", "T−1 h")

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

    if station_details:
        st.markdown("---")
        render_station_geometry_map(station_details, zone_dir.name, include_event=False)
        render_station_metadata_visible(station_details, station_source_label, station_source_files)

    st.markdown("</div>", unsafe_allow_html=True)


def render_quick_guide_outputs() -> None:
    """Render a detailed guide explaining how to read the five monitoring outputs together."""
    st.markdown("---")
    st.markdown(
        """
### How to read these outputs

The system analyzes short time windows across multiple seismic stations simultaneously.

At each moment, it evaluates how the network behaves collectively across space, and how that collective behaviour evolves over time.

Each panel captures a different aspect of this multistation organization:

- **Multistation synchrony**  
  This shows how many stations exhibit coordinated behaviour at the same time. High values indicate that a large fraction of the network is evolving coherently, suggesting a distributed response rather than isolated local activity.

- **Extreme anomaly distribution**  
  This panel highlights when the strongest deviations from typical behaviour occur. It helps identify whether the system exhibits isolated peaks or temporally clustered extremes, which may reflect structured collective dynamics.

- **Mean anomaly and susceptibility**  
  This represents the global evolution of the network response. Sustained increases suggest a gradual build-up of collective activity, while sharp peaks may indicate transient network-wide activation.

- **Station-resolved z-scores**  
  This shows how unusual each station is relative to its own baseline. The key feature is not individual excursions, but whether multiple stations deviate simultaneously, supporting a network-level pattern.

- **Anomaly vs. synthetic tidal forcing**  
  This compares the observed behaviour with a smooth periodic reference signal. It helps distinguish between structured, internally driven responses and externally modulated or smooth variations. This curve is a reference, not a prediction signal.

**Important:** these panels must be interpreted together.

No single graph captures the behaviour of the system on its own. A true multistation signal emerges only when multiple panels show consistent structure at the same time:

- synchrony increases
- anomalies become sustained or clustered
- multiple stations deviate simultaneously

The system is designed to reveal this collective pattern, not isolated features.
        """
    )




def is_baseline_region(raw_name: str) -> bool:
    name = raw_name.lower()
    return any(k in name for k in ["baseline", "low_activity", "low-activity", "nerpio", "minglanilla"])


def clean_zone_display_name(raw_name: str) -> str:
    """Return a human-facing region title without internal RECENT/timestamp tokens."""
    labelled = prettify_zone_name(raw_name)
    if labelled != raw_name.replace("__", " ").replace("_", " ").strip():
        return labelled

    name = raw_name.replace("__", " ").replace("_", " ").replace("-", " ").strip()
    name = re.sub(r"\bRECENT\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b20\d{6}\b", "", name)
    name = re.sub(r"\b\d{6}\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else raw_name


def zone_role_subtitle(raw_name: str) -> str:
    name = raw_name.lower().replace("_", "-")
    if is_baseline_region(raw_name):
        return "Low-activity inland reference · baseline comparison region"
    if "torremolinos" in name or "alboran" in name:
        return "Local southern-Spain reference region"
    if any(k in name for k in ["fault", "subduction", "san-andreas", "cascadia", "hikurangi", "anatolian", "sumatra", "alaska", "kamchatka", "kuril", "peru", "ecuador"]):
        return "Fault / subduction monitoring region"
    if any(k in name for k in ["volcan", "reykjanes", "la-palma", "lapalma", "yellowstone", "etna", "stromboli", "kilauea", "hawaii", "fuego", "popocat"]):
        return "Volcanic / hydrothermal monitoring region"
    return classify_zone(raw_name)


def render_region_role_notice(raw_name: str) -> None:
    if is_baseline_region(raw_name):
        st.info(
            "Low-activity inland reference region used as a baseline for comparison with active seismic, "
            "volcanic and fault/subduction systems. It is processed with the same fixed pipeline and no special tuning."
        )



# ============================================================
# COUNTRY FLAGS FOR REGION BUTTONS
# ============================================================

def get_flag(country_code: str) -> str:
    """Return emoji flag for a two-letter country code.

    Note: some Windows/browser combinations render flag emojis as two-letter
    regional indicators instead of colored flags. The UI avoids duplicating the
    country code so it still remains clean in that case.
    """
    flags = {
        "US": "🇺🇸",
        "JP": "🇯🇵",
        "ES": "🇪🇸",
        "IT": "🇮🇹",
        "MX": "🇲🇽",
        "GT": "🇬🇹",
        "IS": "🇮🇸",
        "NZ": "🇳🇿",
        "CL": "🇨🇱",
        "ID": "🇮🇩",
        "PE": "🇵🇪",
        "RU": "🇷🇺",
        "TR": "🇹🇷",
        "GR": "🇬🇷",
    }
    return flags.get(country_code.upper(), "")


def get_country_code_from_name(raw_name: str) -> str:
    """Infer a country code from the region folder/name.

    This is UI-only. It does not affect the scientific pipeline or analysis.
    """
    name = raw_name.lower().replace("-", "_")

    mapping = {
        # Spain
        "spain": "ES",
        "españa": "ES",
        "nerpio": "ES",
        "lorca": "ES",
        "torremolinos": "ES",
        "minglanilla": "ES",
        "alboran": "ES",
        "la_palma": "ES",
        "lapalma": "ES",

        # USA
        "usa": "US",
        "united_states": "US",
        "yellowstone": "US",
        "san_andreas": "US",
        "sanandreas": "US",
        "cascadia": "US",
        "alaska": "US",
        "aleutian": "US",
        "kilauea": "US",
        "hawaii": "US",

        # Japan
        "japan": "JP",
        "miyako": "JP",
        "tohoku": "JP",

        # Chile
        "chile": "CL",
        "maule": "CL",
        "illapel": "CL",

        # Italy
        "italy": "IT",
        "etna": "IT",
        "stromboli": "IT",

        # Mexico / Guatemala / Iceland
        "mexico": "MX",
        "popocatepetl": "MX",
        "popocat": "MX",
        "guatemala": "GT",
        "fuego": "GT",
        "iceland": "IS",
        "reykjanes": "IS",

        # New Zealand
        "new_zealand": "NZ",
        "newzealand": "NZ",
        "kermadec": "NZ",
        "hikurangi": "NZ",

        # Indonesia
        "indonesia": "ID",
        "bengkulu": "ID",
        "sunda": "ID",
        "sumatra": "ID",
        "andaman": "ID",

        # Peru / Ecuador shown as Peru for compact UI
        "peru": "PE",
        "ecuador": "PE",
        "peru_ecuador": "PE",

        # Russia / Turkey / Greece
        "russia": "RU",
        "kamchatka": "RU",
        "kuril": "RU",
        "turkey": "TR",
        "anatolian": "TR",
        "greece": "GR",
        "aegean": "GR",
    }

    for key, code in mapping.items():
        if key in name:
            return code

    return ""


def compact_region_name_for_button(raw_name: str) -> str:
    """Clean region name for compact button display.

    Removes country codes and long country suffixes to avoid labels like:
        US US Yellowstone, Wyoming, USA
    """
    name = clean_zone_display_name(raw_name)

    replacements = {
        "Torremolinos / Alboran Sea, Spain": "Torremolinos",
        "Nerpio (Albacete, Spain)": "Nerpio",
        "Minglanilla (Spain)": "Minglanilla",
        "Lorca (Murcia, Spain)": "Lorca",
        "Miyako, Japan": "Miyako",
        "Maule, Chile": "Maule",
        "San Andreas Fault, California, USA": "San Andreas",
        "Cascadia Subduction Zone, USA–Canada": "Cascadia",
        "Aegean Sea, Greece–Turkey": "Aegean",
        "Hikurangi Subduction Zone, New Zealand": "Hikurangi",
        "Yellowstone, Wyoming, USA": "Yellowstone",
        "Reykjanes Peninsula, Iceland": "Reykjanes",
        "La Palma, Canary Islands, Spain": "La Palma",
        "Etna, Italy": "Etna",
        "Stromboli, Italy": "Stromboli",
        "Kīlauea, Hawaiʻi, USA": "Kīlauea",
        "Fuego, Guatemala": "Fuego",
        "Popocatépetl, Mexico": "Popocatépetl",
        "Alaska–Aleutian Subduction Zone, USA": "Alaska–Aleutian",
        "Kamchatka–Kuril Subduction Zone, Russia–Japan": "Kamchatka–Kuril",
        "Sumatra–Andaman Subduction Zone, Indonesia": "Sumatra–Andaman",
        "Peru–Ecuador Subduction Zone": "Peru–Ecuador",
        "North Anatolian Fault, Turkey": "North Anatolian",
    }
    if name in replacements:
        return replacements[name]

    # Remove leading country codes that may already be present in folder-derived names.
    name = re.sub(r"^(US|JP|ES|IT|MX|GT|IS|NZ|CL|ID|PE|RU|TR|GR)\s+", "", name, flags=re.IGNORECASE)

    # Remove parenthetical details and common country suffixes.
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    name = re.sub(
        r",?\s*(USA|United States|Spain|Japan|Chile|Italy|Mexico|Guatemala|Iceland|New Zealand|Indonesia|Russia|Turkey|Greece|Peru|Ecuador|Canada)$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip(" ,")

    return name

# ============================================================
# NETWORK SUMMARY / REGION TYPE HELPERS
# ============================================================

REGION_TYPE_STYLE = {
    "Volcanic": {
        "icon": "🌋",
        "label": "Volcanic / hydrothermal",
        "color": "#f97316",
        "subtitle": "Magmatic or hydrothermal monitoring region",
    },
    "Subduction": {
        "icon": "🌊",
        "label": "Subduction / megathrust",
        "color": "#38bdf8",
        "subtitle": "Plate-boundary or megathrust monitoring region",
    },
    "Fault": {
        "icon": "⚡",
        "label": "Fault system",
        "color": "#a78bfa",
        "subtitle": "Crustal fault or tectonic shear-zone region",
    },
    "Earthquake": {
        "icon": "🌐",
        "label": "Seismic region",
        "color": "#22c55e",
        "subtitle": "General seismic monitoring region",
    },
    "Baseline": {
        "icon": "📍",
        "label": "Baseline / reference",
        "color": "#94a3b8",
        "subtitle": "Reference region processed with the same fixed pipeline",
    },
}

NETWORK_STATE_STYLE = {
    "ROBUST COHERENT NETWORK REGIME": ("#ef4444", "Robust coherent regime"),
    "COHERENT NETWORK REGIME": ("#f97316", "Coherent regime"),
    "ROBUST LOCALIZED TRANSIENT SYNCHRONIZATION": ("#f59e0b", "Robust transient synchrony"),
    "LOCALIZED TRANSIENT SYNCHRONIZATION": ("#f59e0b", "Transient synchrony"),
    "TRANSIENT COHERENT ANOMALY": ("#fb7185", "Transient coherent anomaly"),
    "LOCALIZED ANOMALY REGIME": ("#eab308", "Localized anomaly"),
    "ISOLATED EXTREME OUTLIER REGIME": ("#fde047", "Isolated outlier"),
    "CLUSTERED EXTREME EPISODES WITHOUT BROAD SYNCHRONY": ("#60a5fa", "Clustered extremes"),
    "BACKGROUND / LOW COHERENCE": ("#94a3b8", "Background / low coherence"),
}


def physical_region_type(raw_name: str) -> str:
    """Single primary region type for UI badges and filters.

    This is contextual geography only; it does not change the TAMC analysis.
    """
    name = raw_name.lower().replace("-", "_")

    if is_baseline_region(raw_name) or "torremolinos" in name or "alboran" in name:
        return "Baseline"

    if any(k in name for k in [
        "reykjanes", "iceland", "la_palma", "lapalma", "yellowstone", "etna",
        "stromboli", "kilauea", "hawaii", "fuego", "popocatepetl", "popocat",
        "volcan", "volcano", "volcanic"
    ]):
        return "Volcanic"

    if any(k in name for k in [
        "subduction", "megathrust", "tohoku", "miyako", "cascadia", "hikurangi",
        "alaska", "aleutian", "kamchatka", "kuril", "sumatra", "andaman",
        "peru", "ecuador", "maule", "illapel", "kermadec", "bengkulu", "sunda"
    ]):
        return "Subduction"

    if any(k in name for k in [
        "fault", "san_andreas", "sanandreas", "anatolian", "lorca", "aegean",
        "greece", "turkey"
    ]):
        return "Fault"

    return "Earthquake"


def region_type_badge(raw_name: str) -> str:
    rtype = physical_region_type(raw_name)
    style = REGION_TYPE_STYLE.get(rtype, REGION_TYPE_STYLE["Earthquake"])
    return (
        f"<span class='region-type-badge' style='border-color:{style['color']}; "
        f"color:{style['color']}; background:rgba(15,23,42,.78);'>"
        f"{style['icon']} {style['label']}</span>"
    )


def layer_value_for_region_type(rtype: str) -> str:
    return {
        "Volcanic": "Volcanic monitoring",
        "Subduction": "Subduction zones",
        "Fault": "Fault systems",
        "Earthquake": "Earthquake regions",
        "Baseline": "Baseline / reference",
    }.get(rtype, "Earthquake regions")


@st.cache_data(show_spinner=False)
def load_network_summary(zone_dir_str: str) -> Dict[str, object] | None:
    """Load the per-region automatic network interpretation JSON."""
    zone_dir = Path(zone_dir_str)
    candidates = [
        zone_dir / "mainshock" / "network_summary.json",
        zone_dir / "network_summary.json",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def get_network_state_style(state: str | None) -> Tuple[str, str]:
    if not state:
        return "#94a3b8", "Not available"
    for key, value in NETWORK_STATE_STYLE.items():
        if key.lower() == str(state).lower():
            return value
    return "#94a3b8", str(state)



def _level_indicator_html(title: str, value: str, level: str, color: str, fill: int) -> str:
    """Small visual indicator used in the public network summary."""
    fill = max(0, min(100, int(fill)))
    return (
        f"<div class='network-indicator-card'>"
        f"  <div class='network-indicator-title'>{html.escape(title)}</div>"
        f"  <div class='network-indicator-value' style='color:{color};'>{html.escape(value)}</div>"
        f"  <div class='network-indicator-bar'><span style='width:{fill}%; background:{color};'></span></div>"
        f"  <div class='network-indicator-level'>{html.escape(level)}</div>"
        f"</div>"
    )


def _network_indicator_set_html(interp: Dict[str, object]) -> str:
    """Build simple low/moderate/high visual indicators from network_summary.json."""
    coherence = str(interp.get("coherence_level", "not available"))
    anomaly = str(interp.get("anomaly_level", "not available"))
    temporal = str(interp.get("temporal_structure", "not available"))
    extreme = str(interp.get("extreme_structure", "not available"))
    forcing = str(interp.get("forcing_relationship", "not available"))
    robust = str(interp.get("robust_statistical_support", "not available"))

    def style_basic(x: str) -> tuple[str, str, int]:
        xl = x.lower()
        if "high" in xl:
            return "HIGH", "#f59e0b", 92
        if "moderate" in xl:
            return "MODERATE", "#38bdf8", 58
        if "low" in xl:
            return "LOW", "#94a3b8", 26
        return "N/A", "#64748b", 12

    coh_level, coh_color, coh_fill = style_basic(coherence)
    anom_level, anom_color, anom_fill = style_basic(anomaly)

    temporal_l = temporal.lower()
    if "sustained" in temporal_l:
        tmp_level, tmp_color, tmp_fill = "SUSTAINED", "#ef4444", 92
    elif "clustered" in temporal_l:
        tmp_level, tmp_color, tmp_fill = "CLUSTERED", "#f59e0b", 68
    elif "sparse" in temporal_l:
        tmp_level, tmp_color, tmp_fill = "SPARSE", "#60a5fa", 38
    elif "no clear" in temporal_l:
        tmp_level, tmp_color, tmp_fill = "LOW", "#94a3b8", 20
    else:
        tmp_level, tmp_color, tmp_fill = "N/A", "#64748b", 12

    extreme_l = extreme.lower()
    if "strongly" in extreme_l:
        ext_level, ext_color, ext_fill = "STRONG", "#f59e0b", 86
    elif "clustered" in extreme_l:
        ext_level, ext_color, ext_fill = "CLUSTERED", "#38bdf8", 62
    elif "diffuse" in extreme_l or "weak" in extreme_l:
        ext_level, ext_color, ext_fill = "WEAK", "#94a3b8", 30
    else:
        ext_level, ext_color, ext_fill = "N/A", "#64748b", 12

    forcing_l = forcing.lower()
    if "strong" in forcing_l:
        forc_level, forc_color, forc_fill = "STRONG", "#ef4444", 86
    elif "moderate" in forcing_l:
        forc_level, forc_color, forc_fill = "MODERATE", "#38bdf8", 58
    elif "weak" in forcing_l or "no" in forcing_l:
        forc_level, forc_color, forc_fill = "WEAK", "#94a3b8", 26
    else:
        forc_level, forc_color, forc_fill = "N/A", "#64748b", 12

    robust_l = robust.lower()
    if "above_null" in robust_l or "above null" in robust_l:
        rob_level, rob_color, rob_fill = "ABOVE NULL", "#22c55e", 80
    elif "consistent" in robust_l:
        rob_level, rob_color, rob_fill = "NULL-LIKE", "#94a3b8", 28
    else:
        rob_level, rob_color, rob_fill = "N/A", "#64748b", 12

    items = [
        _level_indicator_html("Coherence", coherence.replace("_", " "), coh_level, coh_color, coh_fill),
        _level_indicator_html("Anomaly", anomaly.replace("_", " "), anom_level, anom_color, anom_fill),
        _level_indicator_html("Persistence", temporal.replace("_", " "), tmp_level, tmp_color, tmp_fill),
        _level_indicator_html("Extremes", extreme.replace("_", " "), ext_level, ext_color, ext_fill),
        _level_indicator_html("Forcing", forcing.replace("_", " "), forc_level, forc_color, forc_fill),
        _level_indicator_html("Robust/null", robust.replace("_", " "), rob_level, rob_color, rob_fill),
    ]
    return "<div class='network-indicator-grid'>" + "".join(items) + "</div>"


def render_network_interpretation(zone_dir: Path) -> None:
    """Render automatic network-coherence analysis below the plots.

    Clean UI:
    - Level 1 visible: state, summary, score and tags.
    - Level 2 expandable: ordered cards, no raw HTML and no full JSON block.
    """
    summary = load_network_summary(str(zone_dir.resolve()))
    if not summary:
        st.markdown("---")
        st.info("Network interpretation JSON was not found for this region yet.")
        return

    interp = summary.get("joint_interpretation") or summary.get("interpretation") or {}
    graph_interps = summary.get("graph_interpretations") or {}

    state = str(interp.get("network_state", "Not available"))
    score = interp.get("descriptive_score_0_100")
    score_level = str(interp.get("descriptive_score_level", "not available")).upper()
    summary_text = str(interp.get("summary", "No summary available."))

    state_color, _state_label = get_network_state_style(state)

    indicator_html = _network_indicator_set_html(interp)

    score_txt = "—"
    if score is not None:
        try:
            score_txt = f"{float(score):.2f}"
        except Exception:
            score_txt = str(score)

    st.markdown("---")
    st.markdown("### 🧠 Network coherence interpretation")

    # ========================================================
    # LEVEL 1 — visible, simple, high-level interpretation
    # ========================================================
    st.markdown(
        f"""
<div class="network-interpretation-card level-one-card" style="border-color:{state_color};">
  <div class="network-state-row">
    <div>
      <div class="network-label">Network state</div>
      <div class="network-state" style="color:{state_color};">{html.escape(state)}</div>
      <div class="network-summary">{html.escape(summary_text)}</div>
      <div class="network-indicator-wrap">{indicator_html}</div>
    </div>
    <div class="network-score-box">
      <div class="network-label">Structure score</div>
      <div class="network-score">{html.escape(score_txt)} / 100</div>
      <div class="network-score-level">{html.escape(score_level)}</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # LEVEL 2 — expandable, card-based and ordered
    # ========================================================
    evidence = interp.get("evidence") or []
    consistency = interp.get("cross_plot_consistency") or []

    with st.expander("🔎 Deeper network analysis", expanded=False):
        st.markdown(
            """
<div class="deep-analysis-intro">
  Detailed interpretation generated from the CSV/JSON outputs of the fixed pipeline.
  This section is descriptive only and is not a warning, prediction or risk estimate.
</div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            evidence_items = "".join(
                f"<li>{html.escape(str(item))}</li>"
                for item in evidence
            ) or "<li>No joint evidence listed in JSON.</li>"

            st.markdown(
                f"""
<div class="analysis-card">
  <div class="analysis-card-title">Joint evidence</div>
  <div class="analysis-card-subtitle">What the combined CSV outputs say</div>
  <ul class="analysis-list">{evidence_items}</ul>
</div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            consistency_items = "".join(
                f"<li>{html.escape(str(item))}</li>"
                for item in consistency
            ) or "<li>No cross-plot consistency notes listed in JSON.</li>"

            st.markdown(
                f"""
<div class="analysis-card">
  <div class="analysis-card-title">Cross-plot reading</div>
  <div class="analysis-card-subtitle">How the outputs agree or differ</div>
  <ul class="analysis-list">{consistency_items}</ul>
</div>
                """,
                unsafe_allow_html=True,
            )

        if graph_interps:
            st.markdown("<div class='analysis-section-title'>Graph-by-graph interpretation</div>", unsafe_allow_html=True)

            ordered_keys = [
                "sync_multistation",
                "zscore_multistation",
                "extreme_anomaly_distribution",
                "anomaly_vs_synthetic_tidal_forcing",
                "robust_precursors",
            ]

            graph_items = []
            for key in ordered_keys:
                gi = graph_interps.get(key)
                if gi:
                    graph_items.append((key, gi))

            # Native Streamlit columns avoid the raw HTML rendering bug.
            for i in range(0, len(graph_items), 2):
                cols = st.columns(2)
                for j, (_key, gi) in enumerate(graph_items[i:i + 2]):
                    with cols[j]:
                        title = html.escape(str(gi.get("title", "Graph interpretation")))
                        key_message = html.escape(str(gi.get("key_message", "")))
                        reading = html.escape(str(gi.get("reading", "")))
                        caveat = gi.get("caveat")

                        caveat_html = ""
                        if caveat:
                            caveat_html = f"<div class='analysis-caveat'>{html.escape(str(caveat))}</div>"

                        st.markdown(
                            f"""
<div class="analysis-card graph-analysis-card">
  <div class="analysis-card-title">{title}</div>
  <div class="analysis-card-key">{key_message}</div>
  <div class="analysis-card-body">{reading}</div>
  {caveat_html}
</div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Intentionally not showing the raw network_summary.json here.
        # It was visually noisy in the public interface.
        # The JSON remains available in GitHub/web_data for reproducibility.


def render_zone_detail(zone_dir: Path, show_all: bool) -> None:
    display_name = clean_zone_display_name(zone_dir.name)
    assigned, all_images = load_zone_images(str(zone_dir.resolve()))
    stations, station_details, station_source_files, station_source_label = load_zone_station_metadata(str(zone_dir.resolve()))

    # Region header: applies to every selected zone, not only Maule.
    st.markdown(
        f"""
        <div class="zone-title-card visual-zone-title-card">
            <div class="zone-kicker">SELECTED MONITORING REGION</div>
            <div class="zone-title zone-title-impact">{display_name}</div>
            <div class="zone-type-line">{region_type_badge(zone_dir.name)}</div>
            <div class="zone-subtitle compact-zone-subtitle">
                {format_zone_datetime(zone_dir.name)} · fixed TAMC–FRANJAMAR pipeline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_region_role_notice(zone_dir.name)

    st.markdown("---")
    st.markdown("### Five core outputs")

    cols = st.columns(5)
    for idx, (slot, img) in enumerate(assigned.items()):
        with cols[idx % 5]:
            st.markdown(f"<div class='graph-mini-title'>{slot}</div>", unsafe_allow_html=True)
            if img and Path(img).exists():
                st.image(img, use_container_width=True)
                try:
                    with st.popover("Expand", use_container_width=True):
                        st.markdown(f"### {display_name} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
                except Exception:
                    with st.expander("Expand"):
                        st.markdown(f"### {display_name} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
            else:
                st.info("Missing")

    render_network_interpretation(zone_dir)

    if station_details:
        st.markdown("---")
        render_station_metadata_visible(station_details, station_source_label, station_source_files)

    if show_all:
        st.markdown("---")
        st.markdown("### All detected images")
        extra_cols = st.columns(3)
        for idx, img in enumerate(all_images):
            with extra_cols[idx % 3]:
                st.image(img, caption=Path(img).name, use_container_width=True)
# ============================================================
# BUTTON-BASED REGION SELECTOR
# ============================================================
BUTTON_LAYERS = [
    ("🌍 All", "All regions"),
    ("🌋 Volcanic", "Volcanic monitoring"),
    ("🌊 Subduction", "Subduction zones"),
    ("⚡ Fault", "Fault systems"),
    ("🌐 Earthquake", "Earthquake regions"),
    ("📍 Baseline", "Baseline / reference"),
]


def display_layer_for_zone(raw_name: str) -> str:
    """Human-facing primary layer used by the button selector."""
    return layer_value_for_region_type(physical_region_type(raw_name))


def zone_matches_button_layer(zone_dir: Path, layer: str) -> bool:
    """Layer membership for app buttons.

    Each region has one primary context type. This is visual/contextual only
    and does not affect the fixed TAMC processing pipeline.
    """
    primary = display_layer_for_zone(zone_dir.name)
    if layer == "All regions":
        return True
    return primary == layer


def layer_display_title(layer: str) -> str:
    if layer == "Subduction zones":
        return "Subduction / megathrust zones"
    if layer == "Fault systems":
        return "Fault systems"
    if layer == "Earthquake regions":
        return "Earthquake / seismic regions"
    if layer == "Baseline / reference":
        return "Baseline / reference regions"
    return layer


def layer_button_count(zone_dirs: List[Path], layer: str) -> int:
    return sum(1 for z in zone_dirs if zone_matches_button_layer(z, layer))


def region_button_label(zone_dir: Path) -> str:
    """Compact button label with country flag + region name.

    Example:
        🇪🇸 Nerpio
        🇺🇸 Yellowstone
        🇯🇵 Miyako

    The country code is not repeated because some Windows/browser setups render
    flag emojis as letters. This avoids ugly labels such as "US US Yellowstone".
    """
    country_code = get_country_code_from_name(zone_dir.name)
    flag = get_flag(country_code)
    name = compact_region_name_for_button(zone_dir.name)

    label = f"{flag} {name}".strip() if flag else name
    return label[:42] + "…" if len(label) > 45 else label


def ensure_selector_state(zone_dirs: List[Path]) -> None:
    if "selected_layer" not in st.session_state:
        st.session_state.selected_layer = "All regions"
    valid_layers = [value for _, value in BUTTON_LAYERS]
    if st.session_state.selected_layer not in valid_layers:
        st.session_state.selected_layer = "All regions"

    if "show_region_catalog" not in st.session_state:
        # Clean landing view: START HERE + selected/focused region only.
        # Full region catalogue opens only after pressing "Explore all regions".
        st.session_state.show_region_catalog = False

    if "selected_zone_name" not in st.session_state or not any(z.name == st.session_state.selected_zone_name for z in zone_dirs):
        st.session_state.selected_zone_name = zone_dirs[0].name if zone_dirs else None

    # Only force the selected zone to match the active layer when the catalogue is open.
    # Otherwise the quick reference buttons can keep San Andreas / Yellowstone selected
    # while the catalogue remains hidden.
    if st.session_state.get("show_region_catalog", False):
        filtered = [z for z in zone_dirs if zone_matches_button_layer(z, st.session_state.selected_layer)]
        if filtered and not any(z.name == st.session_state.selected_zone_name for z in filtered):
            st.session_state.selected_zone_name = filtered[0].name


def render_button_selector(zone_dirs: List[Path]) -> Path | None:
    """App-style layer + region selector using large buttons.

    Designed for both desktop and Android WebView: large tap targets,
    no long dropdowns, persistent state, and a simple vertical flow.
    """
    ensure_selector_state(zone_dirs)

    st.markdown("<div class='button-section-title'>MONITORING LAYER</div>", unsafe_allow_html=True)

    visible_layers = []
    for label, layer_value in BUTTON_LAYERS:
        count = layer_button_count(zone_dirs, layer_value)
        if layer_value == "All regions" or count > 0:
            visible_layers.append((label, layer_value, count))

    valid_visible_values = [layer_value for _label, layer_value, _count in visible_layers]
    if st.session_state.selected_layer not in valid_visible_values:
        st.session_state.selected_layer = "All regions"

    layer_cols = st.columns(len(visible_layers))
    for idx, (label, layer_value, count) in enumerate(visible_layers):
        selected = st.session_state.selected_layer == layer_value
        button_label = f"{'✅ ' if selected else ''}{label} · {count}"
        with layer_cols[idx]:
            if st.button(button_label, key=f"layer_btn_{layer_value}", use_container_width=True):
                st.session_state.show_region_catalog = True
                st.session_state.selected_layer = layer_value
                filtered_after = [z for z in zone_dirs if zone_matches_button_layer(z, layer_value)]
                st.session_state.selected_zone_name = filtered_after[0].name if filtered_after else None
                request_scroll("region_selector")
                st.rerun()

    filtered = [z for z in zone_dirs if zone_matches_button_layer(z, st.session_state.selected_layer)]
    if not filtered:
        st.warning("No regions detected for this layer.")
        return None

    st.markdown(
        f"<div class='button-section-title region-title'>REGIONS · {layer_display_title(st.session_state.selected_layer)}</div>",
        unsafe_allow_html=True,
    )

    region_cols = st.columns(4)
    for idx, zone in enumerate(filtered):
        selected = zone.name == st.session_state.selected_zone_name
        label = region_button_label(zone)
        label = f"✅ {label}" if selected else label
        with region_cols[idx % 4]:
            if st.button(label, key=f"region_btn_{zone.name}", use_container_width=True):
                st.session_state.show_region_catalog = True
                st.session_state.selected_zone_name = zone.name
                request_scroll("focused_region")
                st.rerun()

    selected_zone = next((z for z in filtered if z.name == st.session_state.selected_zone_name), filtered[0])
    return selected_zone


def inject_google_analytics() -> None:
    """Inject Google Analytics 4 tag without affecting layout."""
    GA_ID = "G-3B53MZ7P9L"

    st.markdown(
        f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}');
        </script>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>

        .global-radar-card {
            border: 1px solid rgba(56,189,248,.42);
            border-radius: 24px;
            padding: 22px;
            margin: 18px 0 18px 0;
            background: linear-gradient(135deg, rgba(8,47,73,.35), rgba(15,23,42,.96));
            box-shadow: 0 18px 48px rgba(0,0,0,.30);
        }
        .radar-head-row { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:16px; }
        .radar-kicker { color:#38bdf8; font-weight:950; letter-spacing:.16em; font-size:1rem; }
        .radar-main { font-size:1.1rem; font-weight:950; margin-top:8px; }
        .radar-subtitle { color:#cbd5e1; margin-top:8px; line-height:1.45; }
        .radar-tap-hint { color:#bae6fd; margin-top:10px; font-weight:800; }
        .radar-live-pill { white-space:nowrap; border:1px solid rgba(34,197,94,.55); background:rgba(22,101,52,.20); color:#bbf7d0; padding:8px 14px; border-radius:999px; font-weight:950; }
        .radar-layout { display:grid; grid-template-columns:minmax(260px, 1.45fr) minmax(360px, 2.6fr); gap:12px; align-items:stretch; }
        .radar-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(155px, 1fr)); gap:10px; }
        .radar-top-card, .radar-region-card { border:1px solid #334155; border-radius:18px; background:rgba(2,6,23,.55); padding:15px; min-height:150px; }
        .radar-top-card { min-height:220px; background:linear-gradient(135deg, rgba(24,24,27,.88), rgba(15,23,42,.88)); }
        .radar-top-badge { display:inline-block; border:1px solid rgba(251,191,36,.55); color:#fed7aa; background:rgba(120,53,15,.30); border-radius:999px; padding:4px 10px; font-weight:950; letter-spacing:.12em; margin-bottom:16px; }
        .radar-top-name, .radar-region-name { color:#f8fafc; font-weight:950; line-height:1.3; }
        .radar-top-meta, .radar-region-type { margin-top:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:.12em; font-weight:900; }
        .radar-top-score, .radar-score-row { margin-top:12px; color:#f8fafc; font-weight:950; }
        .radar-top-score span { font-size:1.35rem; }
        .radar-top-score small, .radar-score-row small { color:#94a3b8; margin-left:2px; }
        .radar-top-state, .radar-state { margin-top:10px; font-weight:950; line-height:1.35; }
        .radar-button-title { color:#93c5fd; font-weight:900; margin:10px 0 8px 0; }
        .feedback-card { border:1px solid rgba(56,189,248,.35); border-radius:18px; padding:18px; background:rgba(15,23,42,.82); margin:14px 0; line-height:1.55; }
        .bottom-native-nav { margin: 26px 0 90px 0; }
        .bottom-native-nav + div { margin-top: 0; }
        .version-status-card {
            display:flex; flex-wrap:wrap; gap:10px; align-items:center;
            border:1px solid rgba(56,189,248,.38);
            background:linear-gradient(135deg, rgba(15,23,42,.72), rgba(8,47,73,.28));
            border-radius:18px; padding:13px 16px; margin:0 0 18px 0;
            box-shadow:0 14px 34px rgba(0,0,0,.22);
            font-weight:900; letter-spacing:.05em; text-transform:uppercase;
            color:#cbd5e1;
        }
        .version-status-beta { color:#f9a8d4; }
        .version-status-data { color:#7dd3fc; }
        .version-status-version { color:#cbd5e1; }
        .version-separator { color:#64748b; font-weight:950; }
        @media (max-width: 900px) {
            .radar-head-row { flex-direction:column; }
            .radar-layout { grid-template-columns:1fr; }
            .radar-grid { grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }
        }

        .start-here-card {
            border: 1px solid rgba(56,189,248,.22);
            border-radius: 18px;
            padding: 18px 20px;
            margin: 18px 0 14px 0;
            background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.92));
            box-shadow: inset 0 0 0 1px rgba(148,163,184,.04);
        }
        .start-here-text { color: #cbd5e1; font-size: 1.02rem; margin-top: 8px; line-height: 1.45; }
        .radar-card-link { display:block; height:100%; color:inherit !important; text-decoration:none !important; }
        .radar-card-link:hover .clickable-radar-card { border-color:#38bdf8 !important; box-shadow:0 0 0 1px rgba(56,189,248,.35), 0 16px 34px rgba(0,0,0,.38); transform:translateY(-2px); }
        .clickable-radar-card { cursor:pointer; transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease; }

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
        .top-hero h1 { font-size: 2.65rem; margin: 0.1rem 0 0.6rem 0; }
        .top-hero p { color: var(--muted); font-size: 1.18rem; max-width: 850px; }
        .eyebrow { color: var(--accent); font-weight: 800; font-size: .9rem; letter-spacing: .16em; }
        .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .metric { background: rgba(15,23,42,.9); border: 1px solid #25314a; border-radius: 18px; padding: 16px; }
        .metric span { display: block; font-size: 1.85rem; font-weight: 800; color: #f8fafc; }
        .metric small { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
        .update-panel { display: grid; grid-template-columns: 0.8fr 1fr 1fr; gap: 12px; margin: 12px 0 12px 0; }
        .update-box { border: 1px solid #22304a; background: rgba(15,23,42,.86); border-radius: 18px; padding: 14px 16px; box-shadow: 0 10px 28px rgba(0,0,0,.22); }
        .update-main { border-color: rgba(56,189,248,.55); background: linear-gradient(135deg, rgba(14,116,144,.26), rgba(15,23,42,.9)); }
        .update-label { display: block; color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .11em; margin-bottom: 4px; }
        .update-box b { color: #f8fafc; font-size: 1.18rem; }
        .run-strip { border: 1px solid #22304a; background: rgba(8,13,25,.8); padding: 12px 16px; border-radius: 16px; color: #cbd5e1; margin-bottom: 18px; }
        .disclaimer-card { border: 1px solid rgba(249,115,115,.45); background: linear-gradient(135deg, rgba(127,29,29,.35), rgba(15,23,42,.95)); border-radius: 22px; padding: 24px; margin: 12px 0 22px 0; box-shadow: 0 18px 45px rgba(0,0,0,.28); }
        .disclaimer-card h2 { margin-top: 8px; }
        .badge-red { display:inline-block; background: rgba(248,113,113,.15); color:#fecaca; border:1px solid rgba(248,113,113,.45); border-radius: 999px; padding: 6px 11px; font-weight: 800; font-size: .75rem; letter-spacing: .1em; }
        .negative { color: #fecaca; }
        .muted { color: var(--muted); }
        .hero-caption { border: 1px solid #22304a; background: rgba(15,23,42,.75); padding: 10px 14px; border-radius: 14px; margin-bottom: 10px; color: #cbd5e1; }
        .zone-card { border: 1px solid var(--line); background: rgba(15,23,42,.86); border-radius: 22px; padding: 18px; margin-bottom: 16px; box-shadow: 0 12px 30px rgba(0,0,0,.22); }
        .zone-meta { color: var(--muted); border-bottom: 1px solid #1f2937; padding-bottom: 10px; margin-bottom: 14px; }
        .region-context { border: 1px solid #22304a; background: rgba(8,13,25,.78); border-radius: 14px; padding: 12px 14px; color: #cbd5e1; margin: 10px 0 14px 0; font-size: 1.05rem; line-height: 1.45; }
        .region-context b { color: #e0f2fe; }
        .graph-mini-title { min-height: 48px; font-size: .96rem; font-weight: 800; color: #e5e7eb; line-height: 1.15; margin-bottom: 6px; }
        div.stButton > button { width: 100%; border-radius: 12px; border: 1px solid #334155; background: #0f172a; color: #e5e7eb; font-weight: 700; }
        div.stButton > button:hover { border-color: #38bdf8; color: #e0f2fe; }
        [data-testid="stMetric"] { background: rgba(2,6,23,.35); border: 1px solid #22304a; border-radius: 16px; padding: 10px; }
        img { border-radius: 16px; }
        [data-testid="stRadio"] label span { color: #e5e7eb !important; }
        [data-testid="stSelectbox"] label { color: #cbd5e1 !important; font-weight: 700; }
        [data-baseweb="select"] > div { background-color: #f8fafc !important; color: #0f172a !important; }
        [data-baseweb="select"] span { color: #0f172a !important; }
        .selector-card {
            border: 1px solid #22304a;
            background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.88));
            border-radius: 22px;
            padding: 18px 20px;
            margin: 18px 0 20px 0;
            box-shadow: 0 16px 42px rgba(0,0,0,.25);
        }
        .selector-title {
            color: #38bdf8;
            font-size: 1rem;
            letter-spacing: .14em;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .selector-subtitle {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin-bottom: 12px;
        }
        .plain-definition { color: #e0f2fe !important; font-size: 1.32rem !important; margin-bottom: .25rem !important; }
        .compact-disclaimer { padding: 18px 20px; margin: 14px 0 20px 0; }
        .compact-disclaimer h2 { font-size: 1.55rem; margin-bottom: .35rem; }
        .compact-disclaimer details { margin-top: 10px; color: #cbd5e1; }
        .compact-disclaimer summary { cursor: pointer; color: #e0f2fe; font-weight: 800; }

        /* Bigger, more readable typography across Streamlit */
        html, body, .stApp { font-size: 19px !important; }
        .stApp, .stMarkdown, [data-testid="stMarkdownContainer"] { font-size: 1.08rem !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div {
            font-size: 1.08rem !important;
            line-height: 1.65 !important;
        }
        section.main p, section.main li, section.main div, section.main span { font-size: 1.08rem !important; }
        .strong-hook { color: #f8fafc !important; font-size: 1.42rem !important; font-weight: 800; margin-top: 0.4rem !important; }
        .plain-definition { font-size: 1.42rem !important; line-height: 1.45 !important; }
        .top-hero h1 { font-size: 3.05rem !important; line-height: 1.12 !important; }
        .top-hero p { font-size: 1.24rem !important; line-height: 1.55 !important; }
        .eyebrow { font-size: 1rem !important; }
        .metric span { font-size: 2.05rem !important; }
        .metric small { font-size: .9rem !important; }
        .update-box b { font-size: 1.32rem !important; }
        .update-label { font-size: .9rem !important; }
        .run-strip { font-size: 1.18rem !important; line-height: 1.75 !important; }
        .hero-caption { font-size: 1.16rem !important; }
        .region-context { font-size: 1.12rem !important; line-height: 1.6 !important; }
        .graph-mini-title { font-size: 1.08rem !important; }
        [data-testid="stCaptionContainer"] { font-size: 1.05rem !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-size: 1.08rem !important; }
        [data-testid="stTabs"] button p { font-size: 1.02rem !important; }
        [data-testid="stSidebar"] * { font-size: 1.02rem !important; }
        



        .button-section-title {
            color: #38bdf8;
            font-weight: 900;
            letter-spacing: .16em;
            font-size: .95rem;
            margin: 18px 0 8px 0;
            text-transform: uppercase;
        }
        .region-title {
            margin-top: 20px;
        }
        div.stButton > button {
            min-height: 48px;
            border-radius: 14px !important;
            background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(8,13,25,.94)) !important;
            border: 1px solid #2c3b57 !important;
            color: #e5e7eb !important;
            font-weight: 800 !important;
            box-shadow: 0 8px 22px rgba(0,0,0,.18);
        }
        div.stButton > button:hover {
            border-color: #38bdf8 !important;
            color: #e0f2fe !important;
            transform: translateY(-1px);
        }

        .top-app-shell {
            border: 0 !important;
            background: transparent !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 0 18px 0 !important;
            box-shadow: none !important;
        }
        .app-brand-title {
            color:#e0f2fe;
            font-size:1.35rem;
            font-weight:950;
            letter-spacing:.09em;
            margin-top:10px;
            text-transform:uppercase;
        }
        .app-brand-subtitle {
            color:#94a3b8;
            font-size:.92rem;
            font-weight:700;
            letter-spacing:.04em;
            margin-top:2px;
        }
        [data-testid="baseButton-primary"],
        button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7, #1d4ed8) !important;
            border: 1px solid #7dd3fc !important;
            color: #ffffff !important;
            box-shadow: 0 0 22px rgba(56,189,248,.35) !important;
            transform: translateY(-1px);
        }
        [data-testid="baseButton-primary"] p,
        button[kind="primary"] p {
            color: #ffffff !important;
            font-weight: 950 !important;
        }


        .compact-run-strip {
            font-size: 1.05rem !important;
            line-height: 1.55 !important;
        }
        .high-visibility-notice {
            border-color: rgba(248,113,113,.70) !important;
            background: linear-gradient(135deg, rgba(127,29,29,.42), rgba(15,23,42,.97)) !important;
        }
        .high-visibility-notice h2 {
            color: #fecaca !important;
        }
        .top-hero {
            margin-top: 2px !important;
        }

        .zone-title-card {
            border: 1px solid #22304a;
            background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(2,6,23,.92));
            border-radius: 20px;
            padding: 18px 20px;
            margin: 4px 0 18px 0;
            box-shadow: 0 16px 42px rgba(0,0,0,.25);
        }
        .zone-title {
            font-size: 2.75rem;
            font-weight: 950;
            color: #f8fafc;
            letter-spacing: -0.03em;
            line-height: 1.08;
            margin-bottom: 8px;
        }
        .zone-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.45;
        }
        .zone-intro-pro {
            color: #e5e7eb;
            font-size: 1.14rem;
            font-weight: 650;
            line-height: 1.55;
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid rgba(148,163,184,.18);
        }

        .region-type-badge {
            display: inline-block;
            border: 1px solid;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: .86rem;
            font-weight: 900;
            letter-spacing: .04em;
            margin: 4px 0 8px 0;
            text-transform: uppercase;
        }
        .zone-type-line {
            margin: 4px 0 8px 0;
        }
        .network-interpretation-card {
            border: 1px solid #334155;
            background: linear-gradient(135deg, rgba(15,23,42,.97), rgba(2,6,23,.94));
            border-radius: 22px;
            padding: 18px 20px;
            margin: 10px 0 18px 0;
            box-shadow: 0 18px 44px rgba(0,0,0,.26);
        }
        .network-state-row {
            display: grid;
            grid-template-columns: 1.7fr .6fr;
            gap: 18px;
            align-items: center;
        }
        .network-label {
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: .11em;
            font-size: .8rem;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .network-state {
            font-size: 1.65rem;
            font-weight: 950;
            line-height: 1.12;
            letter-spacing: -.02em;
        }
        .network-summary {
            color: #e5e7eb;
            margin-top: 10px;
            line-height: 1.55;
            font-size: 1.05rem;
        }
        .network-score-box {
            border: 1px solid #22304a;
            background: rgba(2,6,23,.48);
            border-radius: 18px;
            padding: 14px;
            text-align: center;
        }
        .network-score {
            color: #f8fafc;
            font-size: 1.4rem;
            font-weight: 950;
        }
        .network-score-level {
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 850;
            margin-top: 4px;
        }
        .pattern-tag-row {
            margin: 10px 0 18px 0;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .pattern-tag {
            display: inline-block;
            border: 1px solid #334155;
            background: rgba(15,23,42,.86);
            color: #cbd5e1;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .82rem;
            font-weight: 800;
        }


        /* ============================================================
           NETWORK INTERPRETATION — CLEAN TWO-LEVEL CARD LAYOUT
        ============================================================ */
        .level-one-card {
            padding: 22px 24px !important;
            margin-top: 14px !important;
            margin-bottom: 18px !important;
        }
        .level-one-card .network-summary {
            max-width: 1150px;
            font-size: 1.14rem !important;
            line-height: 1.65 !important;
            color: #e5e7eb !important;
        }
        .deep-analysis-intro {
            border: 1px solid #22304a;
            background: rgba(8,13,25,.78);
            border-radius: 16px;
            padding: 12px 14px;
            color: #cbd5e1;
            margin: 8px 0 14px 0;
            line-height: 1.55;
            font-weight: 650;
        }
        .analysis-card {
            border: 1px solid #22304a;
            background: linear-gradient(135deg, rgba(15,23,42,.95), rgba(2,6,23,.88));
            border-radius: 18px;
            padding: 16px 18px;
            margin: 10px 0 14px 0;
            box-shadow: 0 12px 32px rgba(0,0,0,.22);
        }
        .analysis-card-title {
            color: #f8fafc;
            font-size: 1.12rem;
            font-weight: 950;
            letter-spacing: -.01em;
            margin-bottom: 4px;
        }
        .analysis-card-subtitle {
            color: #94a3b8;
            font-size: .92rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .analysis-card-key {
            color: #7dd3fc;
            font-weight: 900;
            margin: 8px 0 8px 0;
            line-height: 1.45;
        }
        .analysis-card-body {
            color: #cbd5e1;
            line-height: 1.58;
            font-size: 1rem;
        }
        .analysis-list {
            margin: 8px 0 0 0;
            padding-left: 1.15rem;
        }
        .analysis-list li {
            color: #dbeafe;
            margin-bottom: 10px;
            line-height: 1.55;
        }
        .analysis-section-title {
            color: #38bdf8;
            font-size: .95rem;
            letter-spacing: .14em;
            font-weight: 950;
            text-transform: uppercase;
            margin: 18px 0 8px 0;
        }
        .graph-analysis-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 8px;
        }
        .graph-analysis-card {
            margin: 0 !important;
        }
        .analysis-caveat {
            margin-top: 10px;
            padding: 10px 12px;
            border-left: 3px solid #f59e0b;
            background: rgba(245,158,11,.08);
            color: #fde68a;
            border-radius: 10px;
            font-size: .95rem;
            line-height: 1.45;
        }
        .json-note-card {
            margin-bottom: 8px !important;
        }



        /* ============================================================
           ABOVE-THE-FOLD COMPACT MODE
           Show the monitor and graphs faster; keep text available but reduced.
        ============================================================ */
        .ultra-compact-disclaimer {
            padding: 10px 14px !important;
            margin: 8px 0 12px 0 !important;
            border-radius: 16px !important;
        }
        .ultra-compact-disclaimer .disclaimer-line {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            color: #fecaca;
            font-size: .98rem;
            line-height: 1.35;
        }
        .ultra-compact-disclaimer details {
            margin-top: 6px !important;
        }
        .ultra-compact-disclaimer p {
            margin: 6px 0 !important;
        }
        .top-hero {
            padding: 18px 22px !important;
            grid-template-columns: 1.55fr 1fr !important;
        }
        .top-hero h1 {
            font-size: 2.25rem !important;
            margin-bottom: .35rem !important;
        }
        .top-hero p {
            font-size: 1.04rem !important;
            margin: .15rem 0 !important;
        }
        .metric {
            padding: 11px 14px !important;
            border-radius: 14px !important;
        }
        .metric span {
            font-size: 1.55rem !important;
        }
        .metric small {
            font-size: .72rem !important;
        }
        .update-panel {
            margin: 8px 0 8px 0 !important;
        }
        .update-box {
            padding: 10px 12px !important;
            border-radius: 14px !important;
        }
        .update-box b {
            font-size: 1.02rem !important;
        }
        .run-strip {
            padding: 8px 12px !important;
            font-size: .95rem !important;
            line-height: 1.38 !important;
            margin-bottom: 10px !important;
        }
        .visual-zone-title-card {
            padding: 20px 22px !important;
            margin: 8px 0 14px 0 !important;
            border-radius: 22px !important;
            background:
              radial-gradient(circle at top left, rgba(56,189,248,.10), transparent 35%),
              linear-gradient(135deg, rgba(15,23,42,.98), rgba(2,6,23,.94)) !important;
        }
        .zone-kicker {
            color: #38bdf8;
            font-size: .78rem;
            font-weight: 950;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .zone-title-impact {
            font-size: 3.25rem !important;
            line-height: 1.02 !important;
            color: #f8fafc !important;
            letter-spacing: -.045em !important;
            text-shadow: 0 0 28px rgba(56,189,248,.12);
        }
        .compact-zone-subtitle {
            margin-top: 10px;
            font-size: .98rem !important;
            color: #94a3b8 !important;
        }
        .zone-type-line {
            margin-top: 10px;
        }
        .region-type-badge {
            font-size: 1.03rem !important;
            padding: 8px 14px !important;
        }

        /* ============================================================
           NETWORK INDICATORS — visual low / moderate / high summary
        ============================================================ */
        .network-indicator-wrap {
            margin-top: 16px;
        }
        .network-indicator-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            max-width: 980px;
        }
        .network-indicator-card {
            border: 1px solid #263853;
            background: rgba(2, 6, 23, .42);
            border-radius: 14px;
            padding: 10px 12px;
            box-shadow: inset 0 0 0 1px rgba(148,163,184,.04);
        }
        .network-indicator-title {
            color: #94a3b8;
            font-size: .76rem;
            font-weight: 950;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .network-indicator-value {
            font-size: .92rem;
            font-weight: 900;
            line-height: 1.25;
            min-height: 2.25em;
        }
        .network-indicator-bar {
            width: 100%;
            height: 7px;
            border-radius: 999px;
            background: rgba(148,163,184,.18);
            overflow: hidden;
            margin: 9px 0 6px 0;
        }
        .network-indicator-bar span {
            display: block;
            height: 100%;
            border-radius: 999px;
            box-shadow: 0 0 12px rgba(255,255,255,.16);
        }
        .network-indicator-level {
            color: #cbd5e1;
            font-size: .76rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        /* ============================================================
           MOBILE RESPONSIVE LAYOUT
           Applies only on phones / narrow screens. Desktop remains unchanged.
        ============================================================ */
        @media (max-width: 768px) {

            .zone-title-impact {
                font-size: 2.05rem !important;
            }
            .visual-zone-title-card {
                padding: 16px !important;
            }
            .top-hero {
                padding: 14px !important;
            }

            .network-indicator-grid {
                grid-template-columns: 1fr !important;
                gap: 8px !important;
            }
            .network-indicator-value {
                min-height: auto !important;
            }

            .network-state-row {
                grid-template-columns: 1fr !important;
                gap: 12px !important;
            }
            .graph-analysis-grid {
                grid-template-columns: 1fr !important;
            }
            .analysis-card {
                padding: 14px !important;
                border-radius: 16px !important;
            }
            html, body, .stApp {
                font-size: 16px !important;
            }

            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
            }

            .top-hero {
                grid-template-columns: 1fr !important;
                padding: 18px !important;
                border-radius: 18px !important;
                gap: 14px !important;
                margin-bottom: 12px !important;
            }

            .top-hero h1 {
                font-size: 2rem !important;
                line-height: 1.12 !important;
                margin-bottom: 0.5rem !important;
            }

            .top-hero p,
            .plain-definition,
            .strong-hook {
                font-size: 1rem !important;
                line-height: 1.45 !important;
            }

            .eyebrow {
                font-size: 0.72rem !important;
                letter-spacing: 0.12em !important;
            }

            .metric-grid {
                grid-template-columns: 1fr 1fr !important;
                gap: 8px !important;
            }

            .metric {
                padding: 12px !important;
                border-radius: 14px !important;
            }

            .metric span {
                font-size: 1.45rem !important;
            }

            .metric small {
                font-size: 0.7rem !important;
                line-height: 1.2 !important;
            }

            .update-panel {
                grid-template-columns: 1fr !important;
                gap: 8px !important;
            }

            .update-box {
                padding: 12px !important;
                border-radius: 14px !important;
            }

            .update-box b {
                font-size: 1rem !important;
            }

            .update-label {
                font-size: 0.72rem !important;
            }

            .run-strip,
            .disclaimer-card,
            .zone-card,
            .region-context {
                padding: 14px !important;
                border-radius: 16px !important;
                font-size: 0.95rem !important;
                line-height: 1.45 !important;
            }

            .compact-disclaimer h2 {
                font-size: 1.25rem !important;
            }

            .hero-caption {
                font-size: 0.92rem !important;
                padding: 10px 12px !important;
            }

            .graph-mini-title {
                min-height: auto !important;
                font-size: 0.95rem !important;
                margin-top: 8px !important;
            }

            img {
                border-radius: 10px !important;
            }

            [data-testid="stTabs"] button p {
                font-size: 0.82rem !important;
            }

            .app-brand-title { font-size:1rem !important; margin-top:8px !important; }
            .app-brand-subtitle { font-size:.72rem !important; }

            .zone-title-card {
                padding: 14px 16px !important;
                border-radius: 16px !important;
                margin-bottom: 14px !important;
            }
            .zone-title {
                font-size: 1.9rem !important;
                line-height: 1.12 !important;
            }
            .zone-subtitle {
                font-size: .88rem !important;
                line-height: 1.35 !important;
            }
            .zone-intro-pro {
                font-size: .95rem !important;
                line-height: 1.45 !important;
            }

            [data-testid="stSidebar"] {
                width: 85vw !important;
            }

            [data-testid="stSidebar"] * {
                font-size: 0.92rem !important;
            }

            [data-testid="stMetric"] {
                padding: 8px !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 1rem !important;
            }

            [data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
            }

            [data-testid="stDataFrame"] {
                font-size: 0.8rem !important;
            }
        }

        .compact-hero-caption {
            padding: 8px 12px !important;
            margin-bottom: 8px !important;
            font-size: .98rem !important;
        }


        /* ============================================================
           COMPACT FEATURED OUTPUT
        ============================================================ */
        .compact-feature-card {
            border: 1px solid #22304a;
            background: linear-gradient(135deg, rgba(15,23,42,.90), rgba(2,6,23,.84));
            border-radius: 16px;
            padding: 10px 14px;
            margin: 10px 0 8px 0;
            box-shadow: 0 10px 26px rgba(0,0,0,.18);
        }
        .compact-feature-title {
            color: #38bdf8;
            font-size: .78rem;
            font-weight: 950;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .compact-feature-caption {
            color: #cbd5e1;
            font-size: .98rem;
            font-weight: 750;
        }


        .selector-card {
            padding: 12px 16px !important;
            margin: 12px 0 14px 0 !important;
        }
        .selector-title,
        .button-section-title {
            margin-top: 12px !important;
        }


        /* Country-flag region buttons */
        div.stButton > button {
            line-height: 1.25 !important;
        }


        /* ============================================================
           APP / WEBVIEW CLEAN MODE
           Hide Streamlit chrome: top bar, toolbar, footer, sidebar,
           collapsed sidebar button and "Hosted with Streamlit" badge.
        ============================================================ */
        header {
            visibility: hidden !important;
            display: none !important;
            height: 0px !important;
        }
        [data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
        }
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
            display: none !important;
            visibility: hidden !important;
            width: 0px !important;
            min-width: 0px !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
        }
        button[kind="header"] {
            display: none !important;
            visibility: hidden !important;
        }
        a[href*="streamlit.io"],
        a[href*="streamlit.app"],
        div[class*="viewerBadge"],
        div[class*="stDeployButton"],
        [data-testid="stDeployButton"] {
            display: none !important;
            visibility: hidden !important;
        }
        .block-container {
            padding-top: 0.15rem !important;
            padding-bottom: 1rem !important;
        }
        .main .block-container {
            padding-top: 0.15rem !important;
        }
        section.main {
            padding-top: 0rem !important;
        }
        html, body {
            background: #020617 !important;
        }



        /* ============================================================
           STREAMLIT CLOUD / ANDROID WEBVIEW CLEANUP
           Hide Streamlit menu, footer and hosted badge as far as CSS allows.
           Android WebView also gets a JS cleanup below.
        ============================================================ */
        #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stSidebar"], section[data-testid="stSidebar"],
        [data-testid="collapsedControl"], [data-testid="stAppDeployButton"],
        [data-testid="stDeployButton"], button[kind="header"],
        a[href*="streamlit.io"], a[href*="streamlit.app"],
        div[class*="viewerBadge"], div[class*="ViewerBadge"], div[class*="stDeployButton"],
        div[class*="stAppDeployButton"], div[class*="deployButton"], iframe[title*="streamlit"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
        }
        .stApp { padding-bottom: 0 !important; }
        .block-container { padding-bottom: 0.75rem !important; }
        @media (max-width: 768px) {
            .block-container { padding-bottom: 0.25rem !important; }
            .selector-card { display: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )




def inject_streamlit_badge_remover() -> None:
    """Best-effort cleanup for Streamlit Cloud badge/footer in Android WebView.

    CSS alone is not always enough because Streamlit Cloud may inject the badge
    after the page has loaded. This small component repeatedly removes only
    Streamlit chrome elements, without touching the dashboard content.
    """
    components.html(
        """
        <script>
        (function() {
            function hideStreamlitChrome() {
                try {
                    const selectors = [
                        '#MainMenu',
                        'footer',
                        'header',
                        '[data-testid="stToolbar"]',
                        '[data-testid="stDecoration"]',
                        '[data-testid="stStatusWidget"]',
                        '[data-testid="stAppDeployButton"]',
                        '[data-testid="stDeployButton"]',
                        '[data-testid="collapsedControl"]',
                        'button[kind="header"]',
                        'a[href*="streamlit.io"]',
                        'a[href*="streamlit.app"]',
                        'div[class*="viewerBadge"]',
                        'div[class*="ViewerBadge"]',
                        'div[class*="stDeployButton"]',
                        'div[class*="stAppDeployButton"]',
                        'div[class*="deployButton"]'
                    ];

                    const doc = window.parent.document;
                    selectors.forEach(function(sel) {
                        doc.querySelectorAll(sel).forEach(function(el) {
                            el.style.setProperty('display', 'none', 'important');
                            el.style.setProperty('visibility', 'hidden', 'important');
                            el.style.setProperty('opacity', '0', 'important');
                            el.style.setProperty('pointer-events', 'none', 'important');
                            el.style.setProperty('height', '0px', 'important');
                            el.style.setProperty('min-height', '0px', 'important');
                            el.style.setProperty('max-height', '0px', 'important');
                            el.style.setProperty('overflow', 'hidden', 'important');
                        });
                    });

                    // Last-resort cleanup for the red hosted badge text.
                    doc.querySelectorAll('body *').forEach(function(el) {
                        const txt = (el.innerText || '').trim();
                        if (txt === 'Hosted with Streamlit' || txt === 'Created with Streamlit') {
                            el.style.setProperty('display', 'none', 'important');
                            if (el.parentElement) {
                                el.parentElement.style.setProperty('display', 'none', 'important');
                            }
                        }
                    });
                } catch (e) {}
            }

            hideStreamlitChrome();
            setTimeout(hideStreamlitChrome, 300);
            setTimeout(hideStreamlitChrome, 1000);
            setTimeout(hideStreamlitChrome, 2500);
            setInterval(hideStreamlitChrome, 2000);
        })();
        </script>
        """,
        height=0,
        width=0,
    )

# ============================================================
# SMOOTH SCROLL / ANCHOR HELPERS
# ============================================================
def request_scroll(target_id: str) -> None:
    """Request a smooth scroll after the next Streamlit rerun."""
    st.session_state.scroll_target = target_id


def perform_deferred_scroll(default_delay_ms: int = 350) -> None:
    """Execute pending smooth scroll once anchors have been rendered."""
    target = st.session_state.get("scroll_target")
    if not target:
        return
    st.session_state.scroll_target = None
    safe_target = re.sub(r"[^A-Za-z0-9_-]", "", str(target))
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            const parentDoc = window.parent.document;
            const el = parentDoc.getElementById('{safe_target}');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
            }}
        }}, {int(default_delay_ms)});
        </script>
        """,
        height=0,
        width=0,
    )


def render_back_to_top_button() -> None:
    """Bottom-centred fixed Inicio button.

    Only the Inicio button is kept at the bottom to avoid duplicating the
    top Monitor navigation. It is injected into the parent document so it
    remains floating while scrolling.
    """
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const oldFeedback = doc.getElementById('franjamarFeedbackBtn');
            if (oldFeedback) oldFeedback.remove();
            const oldMonitor = doc.getElementById('franjamarBottomMonitorBtn');
            if (oldMonitor) oldMonitor.remove();

            let bar = doc.getElementById('franjamarBottomNav');
            if (!bar) {
                bar = doc.createElement('div');
                bar.id = 'franjamarBottomNav';
                doc.body.appendChild(bar);
            }

            let top = doc.getElementById('franjamarBottomTopBtn');
            if (!top) {
                top = doc.createElement('button');
                top.id = 'franjamarBottomTopBtn';
                top.innerHTML = '⬆ Inicio';
                bar.appendChild(top);
            }
            top.onclick = function() {
                // Inicio is a scroll-to-top helper. It never hides the top Monitor button
                // and it does not inject any extra Monitor button at the bottom.
                const el = doc.getElementById('app_top_anchor');
                if (el) {
                    el.scrollIntoView({behavior: 'smooth', block: 'start'});
                } else {
                    window.parent.scrollTo({top: 0, behavior: 'smooth'});
                }
            };

            const styleId = 'franjamarBottomNavStyle';
            let style = doc.getElementById(styleId);
            if (!style) {
                style = doc.createElement('style');
                style.id = styleId;
                doc.head.appendChild(style);
            }
            style.textContent = `
                #franjamarBottomNav {
                    position: fixed !important;
                    left: 50% !important;
                    bottom: 18px !important;
                    transform: translateX(-50%) !important;
                    z-index: 2147483647 !important;
                    display: flex !important;
                    gap: 10px !important;
                    align-items: center !important;
                    justify-content: center !important;
                    padding: 7px !important;
                    border-radius: 999px !important;
                    border: 1px solid rgba(56,189,248,.38) !important;
                    background: rgba(2,6,23,.72) !important;
                    backdrop-filter: blur(10px) !important;
                    box-shadow: 0 12px 34px rgba(0,0,0,.55) !important;
                    pointer-events: auto !important;
                }
                #franjamarBottomNav button {
                    min-width: 118px !important;
                    padding: 10px 16px !important;
                    border-radius: 999px !important;
                    border: 1px solid rgba(125,211,252,.75) !important;
                    background: linear-gradient(135deg, rgba(14,165,233,.98), rgba(37,99,235,.98)) !important;
                    color: white !important;
                    font-weight: 900 !important;
                    font-size: 13px !important;
                    letter-spacing: .03em !important;
                    cursor: pointer !important;
                    box-shadow: 0 8px 22px rgba(0,0,0,.35) !important;
                }
                #franjamarBottomNav button:hover {
                    transform: translateY(-1px) !important;
                    box-shadow: 0 0 18px rgba(56,189,248,.35) !important;
                }
                @media (max-width: 700px) {
                    #franjamarBottomNav { bottom: 10px !important; gap: 7px !important; padding: 6px !important; }
                    #franjamarBottomNav button { min-width: 92px !important; padding: 9px 11px !important; font-size: 12px !important; }
                }
            `;
        })();
        </script>
        """,
        height=0,
        width=0,
    )

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="TAMC–FRANJAMAR Monitor", layout="wide", initial_sidebar_state="collapsed")
hide_streamlit_chrome()
st.markdown("<div id='app_top_anchor' style='height:0; margin:0; padding:0; overflow:hidden;'></div>", unsafe_allow_html=True)
inject_google_analytics()
inject_css()

st.markdown(
    """
    <style>
    .block-container { padding-top: 0rem !important; margin-top: 0rem !important; padding-bottom: 7.5rem !important; }
    header, [data-testid="stHeader"] { display:none !important; height:0 !important; min-height:0 !important; visibility:hidden !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main { padding-top:0 !important; margin-top:0 !important; }
    .top-app-shell {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 0 18px 0 !important;
    }
    .app-brand-title { margin-top: 0 !important; }
    .event-context-card {
        margin: 1.0rem 0 1.2rem 0;
        padding: 1.05rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(56,189,248,0.45);
        background: linear-gradient(135deg, rgba(8,47,73,0.45), rgba(15,23,42,0.95));
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
    }
    .event-context-kicker { color:#38bdf8; font-weight:900; letter-spacing:.16em; font-size:.78rem; }
    .event-context-title { color:#e5e7eb; font-weight:900; font-size:1.25rem; margin-top:.35rem; }
    .event-context-body { color:#cbd5e1; line-height:1.55; margin-top:.4rem; }
    .event-context-note { color:#93c5fd; margin-top:.65rem; font-size:.88rem; font-weight:700; }
    a.ranking-open-link,
    a.radar-open-link {
        display: block;
        width: 100%;
        text-align: center;
        padding: 0.62rem 0.8rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(96,165,250,0.45);
        background: rgba(15,23,42,0.82);
        color: #e5e7eb !important;
        text-decoration: none !important;
        font-weight: 800;
    }
    a.ranking-open-link:hover,
    a.radar-open-link:hover {
        border-color: rgba(56,189,248,0.85);
        color: #bae6fd !important;
        background: rgba(14,165,233,0.16);
    }
    @media (max-width: 700px) {
        .block-container { padding-bottom: 8.5rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_streamlit_badge_remover()

# Public/app mode: no Streamlit sidebar controls.
# The dashboard always reads the fixed web_data/latest folder.
resultados_dir = get_resultados_dir()
auto_refresh = False
refresh_seconds = 60
show_all = False

if resultados_dir is None or not resultados_dir.exists():
    st.error("The fixed results folder web_data/latest was not found.")
    st.stop()

zone_dirs = sorted(list_zone_dirs(resultados_dir), key=zone_sort_key, reverse=True)
if not zone_dirs:
    st.warning("No zone subfolders were found inside the results folder.")
    st.stop()

# ============================================================
# TOP APP NAVIGATION
# ============================================================
def ensure_page_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "monitor"


def go_monitor_home() -> None:
    """Go to the monitor landing screen and jump to START HERE.

    The top Monitor button is kept as a native Streamlit button.
    Its only job is to return to the monitor page and scroll down to the
    START HERE block, which is the practical entry point for users.
    """
    st.session_state.page = "monitor"
    st.session_state.show_region_catalog = False
    st.session_state.show_full_ranking = False
    st.session_state.selected_layer = "All regions"
    if zone_dirs:
        st.session_state.selected_zone_name = zone_dirs[0].name

    # Clear query params robustly across Streamlit versions.
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass

    # Important: this is intentionally START HERE, not the top of the page.
    request_scroll("start_here")


def scroll_to_top() -> None:
    request_scroll("app_top_anchor")


def nav_button(label: str, page_key: str) -> None:
    selected = st.session_state.page == page_key
    final_label = ("✅ " if selected else "") + label

    if page_key == "monitor":
        # Keep Monitor as a real native Streamlit button so it is always visible
        # in the top navigation bar. It resets the app state to the public
        # monitor landing view instead of using an external link.
        if st.button(
            final_label,
            key="nav_monitor_home_button",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            go_monitor_home()
            st.rerun()
        return

    if st.button(
        final_label,
        key=f"nav_{page_key}",
        use_container_width=True,
        type="primary" if selected else "secondary",
    ):
        st.session_state.page = page_key
        st.session_state.show_full_ranking = False
        request_scroll(f"page_{page_key}")
        st.rerun()


def render_top_app_bar() -> None:
    """Top app bar with visible icon and main navigation.

    It is rendered before the hero/header, so the navigation is visible
    immediately on desktop and Android-style WebView.
    """
    ensure_page_state()

    st.markdown("<div class='top-app-shell'>", unsafe_allow_html=True)

    if APP_ICON_PATH.exists():
        icon_col, title_col = st.columns([0.08, 0.92])
        with icon_col:
            st.image(str(APP_ICON_PATH), width=74)
        with title_col:
            st.markdown(
                """
                <div class='app-brand-title'>TAMC–FRANJAMAR Monitor</div>
                <div class='app-brand-subtitle'>Network coherence · fixed pipeline · descriptive monitoring</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class='app-brand-title'>TAMC–FRANJAMAR Monitor</div>
            <div class='app-brand-subtitle'>Network coherence · fixed pipeline · descriptive monitoring</div>
            """,
            unsafe_allow_html=True,
        )

    # Extra vertical separation so Monitor aligns below the subtitle, not over it.
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    nav_cols = st.columns(8)
    with nav_cols[0]:
        nav_button("📡 Monitor", "monitor")
    with nav_cols[1]:
        nav_button("🌍 Event analysis", "events")
    with nav_cols[2]:
        nav_button("📘 What is this", "what")
    with nav_cols[3]:
        nav_button("⚙️ How it works", "how")
    with nav_cols[4]:
        nav_button("📊 Read outputs", "read")
    with nav_cols[5]:
        nav_button("📄 DOI / Papers", "papers")
    with nav_cols[6]:
        nav_button("🚀 Roadmap", "roadmap")
    with nav_cols[7]:
        nav_button("💬 Feedback", "feedback")

    st.markdown("</div>", unsafe_allow_html=True)


# Backward-compatible name in case older code calls it.
def render_app_navigation() -> None:
    render_top_app_bar()


def render_monitor_page() -> None:
    st.markdown("<div id='page_monitor'></div>", unsafe_allow_html=True)
    ensure_selector_state(zone_dirs)
    render_global_status_radar(zone_dirs)
    render_start_here_block(zone_dirs)

    selected_zone = next(
        (z for z in zone_dirs if z.name == st.session_state.selected_zone_name),
        zone_dirs[0],
    )

    # The catalogue stays hidden on the landing view. It only opens after pressing
    # "Explore all regions", keeping START HERE clean and avoiding a long first screen.
    if st.session_state.get("show_region_catalog", False):
        st.markdown("<div id='region_selector'></div>", unsafe_allow_html=True)
        selected_from_selector = render_button_selector(zone_dirs)
        if selected_from_selector is not None:
            selected_zone = selected_from_selector

    st.markdown("---")
    st.markdown("<div id='focused_region'></div>", unsafe_allow_html=True)
    render_zone_detail(selected_zone, show_all=show_all)

    perform_deferred_scroll()



def real_event_button_label(raw_name: str) -> str:
    """Compact label for retrospective event buttons."""
    name = raw_name.lower().replace('-', '_')
    if 'miyako' in name or '20260420' in name:
        return 'Miyako (2026, M7.4)'
    if 'maule' in name or '20100227' in name:
        return 'Maule (2010, M8.8)'
    if 'tohoku' in name or '20110311' in name:
        return 'Tohoku (2011, M9.1)'
    if 'kamchatka' in name or '20250729' in name:
        return 'Kamchatka–Kuril (2025, M8.8)'
    return compact_region_name_for_button(raw_name)

def render_real_events_page() -> None:
    st.markdown("<div id='page_events'></div>", unsafe_allow_html=True)
    st.markdown("## 🌍 Real event analysis")
    st.markdown(
        "Fixed-pipeline retrospective analyses of confirmed earthquake events. "
        "This section is separated from the near-real-time monitor so daily updates cannot overwrite it."
    )
    st.info("These cases use the same multistation framework. They are descriptive analyses, not prediction, warning or risk estimation.")

    event_dir = get_event_analysis_dir()
    if event_dir is None:
        st.warning("The folder `event_analysis` was not found in the repository.")
        return

    event_dirs = sorted(list_zone_dirs(event_dir), key=lambda p: p.name.lower())
    if not event_dirs:
        st.warning("No event case folders were found inside `event_analysis`.")
        return

    if "selected_event_name" not in st.session_state or not any(p.name == st.session_state.selected_event_name for p in event_dirs):
        st.session_state.selected_event_name = event_dirs[0].name

    st.markdown("### Select a real event")
    cols = st.columns(4)
    for idx, ev in enumerate(event_dirs):
        label = real_event_button_label(ev.name)
        flag = get_flag(get_country_code_from_name(ev.name))
        active = ev.name == st.session_state.selected_event_name
        with cols[idx % 4]:
            if st.button(("✅ " if active else "") + f"{flag} {label}".strip(), key=f"event_select_{ev.name}", use_container_width=True):
                st.session_state.selected_event_name = ev.name
                request_scroll("selected_real_event")
                st.rerun()

    selected_event = next((p for p in event_dirs if p.name == st.session_state.selected_event_name), event_dirs[0])
    st.markdown("---")
    st.markdown("<div id='selected_real_event'></div>", unsafe_allow_html=True)
    render_event_detail(selected_event)
    perform_deferred_scroll()


def render_event_detail(event_dir: Path) -> None:
    display_name = clean_zone_display_name(event_dir.name)
    assigned, all_images = load_zone_images(str(event_dir.resolve()))
    stations, station_details, station_source_files, station_source_label = load_zone_station_metadata(str(event_dir.resolve()))

    st.markdown(
        f"""
        <div class="zone-title-card visual-zone-title-card">
            <div class="zone-kicker">SELECTED REAL EVENT ANALYSIS</div>
            <div class="zone-title zone-title-impact">{html.escape(display_name)}</div>
            <div class="zone-type-line">{region_type_badge(event_dir.name)}</div>
            <div class="zone-subtitle compact-zone-subtitle">
                {format_zone_datetime(event_dir.name)} · fixed TAMC–FRANJAMAR pipeline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_event_context_block(event_dir.name)

    st.markdown("### Five core outputs")
    cols = st.columns(5)
    for idx, (slot, img) in enumerate(assigned.items()):
        with cols[idx % 5]:
            st.markdown(f"<div class='graph-mini-title'>{slot}</div>", unsafe_allow_html=True)
            if img and Path(img).exists():
                st.image(img, use_container_width=True)
                try:
                    with st.popover("Expand", use_container_width=True):
                        st.markdown(f"### {display_name} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
                except Exception:
                    with st.expander("Expand"):
                        st.markdown(f"### {display_name} · {slot}")
                        st.caption(GRAPH_SLOTS[slot]["description"])
                        st.image(img, caption=Path(img).name, use_container_width=True)
                        if station_details:
                            render_station_metadata(station_details, station_source_label, station_source_files)
            else:
                st.info("Missing")

    render_network_interpretation(event_dir)

    if station_details:
        st.markdown("---")
        render_station_geometry_map(station_details, event_dir.name, include_event=True)
        render_station_metadata_visible(station_details, station_source_label, station_source_files)


def render_what_page() -> None:
    st.markdown("<div id='page_what'></div>", unsafe_allow_html=True)
    st.markdown("## What is this?")
    st.markdown(f"""
This is an **experimental scientific exploration system** based on the **TAMC–FRANJAMAR** framework.

It studies how seismic networks behave **as a collective system**, rather than treating each station as an isolated signal.

The core idea is simple:

> the relevant structure is not a single peak at one station, but the emergence of coordinated behaviour across multiple stations over time.

The dashboard is directly based on the reproducible research record:

**“TAMC–FRANJAMAR v3: A retrospective and reproducible framework for the analysis of collective statistical behavior in multistation seismic networks”**  
[{SEISMIC_DOI}]({SEISMIC_DOI})

All monitored regions are processed with the same fixed pipeline, using identical parameters and no region-specific tuning. This allows direct comparison between earthquake regions, volcanic systems, fault systems, subduction zones and low-activity baselines.

The app operates in **monitoring mode** using rolling **24 h windows** with a short consolidation delay (**T−1 h**). It generates plots, JSON/CSV summaries and descriptive network-state classifications from the same reproducible output structure.

### What this is not

This is **not** an earthquake prediction app, not an eruption prediction app, not an early-warning system and not an operational risk platform.

It does **not** estimate event timing, magnitude, location or risk.

### What it is for

It is designed for scientific exploration: to characterize statistical structure, detect deviations from baseline behaviour, compare regions under a fixed pipeline and study how distributed geophysical systems organize as a whole.
""")
    st.info("Focus: collective behaviour across stations, not individual amplitudes alone.")


def render_how_page() -> None:
    st.markdown("<div id='page_how'></div>", unsafe_allow_html=True)
    st.markdown("## How it works")
    st.markdown("""
All regions are processed using the exact same fixed pipeline described in the **TAMC–FRANJAMAR v3** framework, with identical parameters and no region-specific tuning.

The system does not analyze stations independently. Instead, it evaluates how multiple stations evolve together over time, focusing on the emergence of coordinated behaviour across the network.

The pipeline transforms raw seismic signals into standardized, comparable responses and then quantifies how these responses organize collectively.

Core processing steps include:

- robust per-station normalization
- aggregation of standardized responses across the network
- detection of multistation exceedances and collective activation
- quantification of synchrony and temporal organization
- comparison against control windows and null-model expectations

The framework can be used in two complementary modes:

- **retrospective analysis mode**: large-scale analysis of more than 200 real earthquakes, using fixed **T−24 h windows** under the same pipeline
- **monitoring mode**: continuous 24 h window with a **T−1 h delay**

This dashboard operates in **monitoring mode**.

Outputs are strictly descriptive and are not used for prediction.

The five diagnostic panels must be interpreted together. The signal of interest is not contained in any single plot, but in the collective multistation pattern.
    """)
    st.warning("The diagnostic value comes from the joint multistation pattern, not from isolated graphs interpreted one by one.")


def render_read_outputs_page() -> None:
    st.markdown("<div id='page_read'></div>", unsafe_allow_html=True)
    render_quick_guide_outputs()




def render_support_callout(compact: bool = False) -> None:
    """Small non-invasive support block linked to Buy Me a Coffee."""
    if compact:
        st.markdown(
            f"""
<style>
.support-callout-card {{
    margin: 1.0rem 0 1.25rem 0;
    padding: 1.05rem 1.15rem;
    border-radius: 18px;
    border: 1px solid rgba(244,114,182,0.48);
    background: linear-gradient(135deg, rgba(83,13,55,0.24), rgba(8,47,73,0.24), rgba(15,23,42,0.94));
    box-shadow: 0 0 0 1px rgba(56,189,248,0.08), 0 14px 30px rgba(0,0,0,0.24);
}}
.support-callout-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 1rem;
    align-items: center;
}}
.support-callout-kicker {{
    color:#f9a8d4;
    font-size:0.78rem;
    font-weight:900;
    letter-spacing:0.14em;
    text-transform:uppercase;
    margin-bottom:0.35rem;
}}
.support-callout-title {{
    color:#ffffff;
    font-weight:900;
    font-size:1.02rem;
    margin-bottom:0.35rem;
}}
.support-callout-text {{
    color:rgba(226,232,240,0.90);
    line-height:1.55;
    font-size:0.94rem;
    max-width: 980px;
}}
.support-callout-note {{
    margin-top:0.45rem;
    color:#bae6fd;
    font-size:0.84rem;
    font-weight:700;
}}
.support-callout-button {{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    white-space:nowrap;
    padding:0.68rem 1.05rem;
    border-radius:999px;
    border:1px solid rgba(251,207,232,0.72);
    background:linear-gradient(135deg, rgba(236,72,153,0.28), rgba(14,165,233,0.18));
    color:#fce7f3 !important;
    text-decoration:none !important;
    font-weight:900;
    box-shadow:0 0 22px rgba(236,72,153,0.20);
}}
.support-callout-button:hover {{
    border-color:rgba(255,255,255,0.95);
    color:#ffffff !important;
    box-shadow:0 0 32px rgba(236,72,153,0.36);
}}
@media (max-width: 900px) {{
    .support-callout-grid {{ grid-template-columns:1fr; }}
    .support-callout-button {{ width:100%; }}
}}
</style>
<div class="support-callout-card">
  <div class="support-callout-grid">
    <div>
      <div class="support-callout-kicker">Independent research</div>
      <div class="support-callout-title">☕ Support this project</div>
      <div class="support-callout-text">
        This is an independent research project fully developed by a single author.
	It includes the scientific paper, the reproducible pipeline, the web application, and the Android app.
	The monitor you see here is only one part of a broader system designed to study collective behaviour in seismic networks under a fixed and reproducible 	framework.

If you find it useful or interesting, you can support its development, infrastructure, and future improvements.
      </div>
      <div class="support-callout-note">Descriptive research · fixed pipeline · no forecasting claims</div>
    </div>
    <a class="support-callout-button" href="{BUYMEACOFFEE_URL}" target="_blank" rel="noopener noreferrer">☕ Support / Buy Me a Coffee</a>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("### 🤝 Support the project")
    st.write(
        "This is a personal research project, developed independently and without funding. "
        "The results shown are generated using the project's reproducible pipeline. "
        "If you find this work useful or interesting, you can support its development and evolution, "
        "helping to implement upcoming improvements."
    )
    st.link_button("☕ Support the project", BUYMEACOFFEE_URL, use_container_width=True)

def render_roadmap_page() -> None:
    st.markdown("<div id='page_roadmap'></div>", unsafe_allow_html=True)
    st.markdown("## 🚀 Future roadmap / Hoja de ruta")
    st.markdown(
        """
<div class="roadmap-card">
  <div class="roadmap-item"><b>Real earthquake and eruption event overlays</b><br>Direct comparison between detected network structure and confirmed events.</div>
  <div class="roadmap-item"><b>Multiregion comparison mode</b><br>Identify similarities and differences between seismic, volcanic, fault and subduction systems.</div>
  <div class="roadmap-item"><b>Temporal evolution viewer</b><br>Follow how network structure evolves across time instead of only reading a single monitoring window.</div>
  <div class="roadmap-item"><b>Additional sensor integration</b><br>Extend the same multistation-structure approach to geomagnetic, atmospheric and other distributed sensor networks.</div>
  <div class="roadmap-item"><b>Improved interpretation layer</b><br>Clearer classification of network regimes, with more transparent explanations for non-specialist users.</div>
  <div class="roadmap-item"><b>Native mobile app version</b><br>Faster access, better mobile navigation and improved near-real-time interaction.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("This roadmap is directional and descriptive. It does not imply operational warning, prediction or risk-estimation functionality.")
    st.markdown("---")
    render_support_callout(compact=False)


def render_feedback_page() -> None:
    st.markdown("<div id='page_feedback'></div>", unsafe_allow_html=True)
    st.markdown("## Feedback / Suggestions")
    st.markdown(
        """
<div class="feedback-card">
  <b>Help improve the app</b><br>
  This is an independent, experimental and non-predictive research app. Feedback is especially useful on usability, scientific interpretation, unclear labels, missing regions, Android display issues, or bugs.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Contact email:**")
    st.code(FEEDBACK_EMAIL, language="text")
    mailto = f"mailto:{FEEDBACK_EMAIL}?subject={quote(FEEDBACK_SUBJECT)}"
    st.markdown(f"[📩 Send feedback by email]({mailto})")
    st.markdown("### Suggested feedback format")
    st.code(
        "App section:\nWhat confused me:\nWhat worked well:\nDevice/browser:\nSuggested region or improvement:",
        language="text",
    )

def render_papers_page() -> None:
    st.markdown("<div id='page_papers'></div>", unsafe_allow_html=True)
    st.markdown("## DOI / Papers")
    st.markdown(f"""
This dashboard is built around reproducible TAMC–FRANJAMAR research records. Each record includes downloadable code, configuration files and documentation.

### Seismic framework and monitoring implementation  
[{SEISMIC_DOI}]({SEISMIC_DOI})

Main reproducible implementation used for earthquake-style and regional monitoring runs. It includes station-selection logic and monitoring mode using a **T−1 h delay**.

### Volcanic framework and eruption analyses  
[{VOLCANIC_DOI}]({VOLCANIC_DOI})

Application of the same fixed multistation framework to volcanic systems such as Etna, Stromboli, Kīlauea, Reykjanes and La Palma-style monitoring contexts.

### Multistation coupling regimes / extreme-event structure  
[{EXTREME_EVENT_DOI}]({EXTREME_EVENT_DOI})

Comparative framework for interpreting events by their collective multistation coupling structure rather than only by their external label.

### Reproducibility statement

All monitoring outputs shown here are direct applications of the same fixed-parameter multistation pipeline: **no region-specific tuning, same 24 h window, same T−1 h monitoring delay**.
    """)


ensure_page_state()
apply_query_params_to_state(zone_dirs)

render_top_app_bar()
render_header(zone_dirs)
render_version_status_card()

if st.session_state.page == "monitor":
    render_support_callout(compact=True)
    render_monitor_page()
elif st.session_state.page == "events":
    render_real_events_page()
elif st.session_state.page == "what":
    render_what_page()
elif st.session_state.page == "how":
    render_how_page()
elif st.session_state.page == "read":
    render_read_outputs_page()
elif st.session_state.page == "roadmap":
    render_roadmap_page()
elif st.session_state.page == "feedback":
    render_feedback_page()
elif st.session_state.page == "papers":
    render_papers_page()

if st.session_state.page != "monitor":
    perform_deferred_scroll()

render_back_to_top_button()

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
