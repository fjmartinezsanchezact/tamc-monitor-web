#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_event_datos.py

Igual que run_event.py, pero ejecuta el pipeline TAMC 24h
para TODOS los directorios que encuentre dentro de ./data/

Uso:
  python run_event_datos.py

Notas:
- Detecta eventos como carpetas dentro de data/
- Excluye ciertas carpetas (configurable)
- Ejecuta los mismos pasos que run_event.py
- NO pasa --data-dir (para ser equivalente a run_event.py)
"""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from typing import List


# =========================
# CONFIGURACIÓN
# =========================
EXCLUDE_DIRS = {
    "last_24h",
    "last_48h_tminus3",
    # si algún día creas subcarpetas auxiliares dentro de data, ponlas aquí
}

PIPELINE_STEPS: List[str] = [
    "00_preprocess_24h.py",
    "01_rotation_3c.py",
    "02_tamc_metrics.py",
    "03_scan_criticality.py",
    "04_sync_multistation.py",
    "05_plotting.py",
    "07_mareas_forzantes.py",
]


# =========================
# UTILIDADES (copiadas de run_event.py)
# =========================
def project_root_from_core(core_dir: Path) -> Path:
    # core_dir = .../src/core
    return core_dir.parent.parent  # .../ (raíz del proyecto)


def ensure_events_csv(project_root: Path) -> None:
    """
    Garantiza que src/events.csv exista.
    Si no existe, lo genera llamando a src/build_events_csv.py.
    """
    events_csv = project_root / "src" / "events.csv"
    builder = project_root / "src" / "build_events_csv.py"

    if events_csv.exists():
        print("[INFO] events.csv encontrado.")
        return

    if not builder.exists():
        print("[WARN] No existe src/build_events_csv.py -> no se puede generar events.csv")
        return

    print("[INFO] events.csv no existe. Generándolo automáticamente (IRIS/FDSN)...")
    cmd = [sys.executable, str(builder)]
    res = subprocess.run(cmd, cwd=str(project_root))

    if res.returncode != 0:
        print("[WARN] No se pudo generar events.csv. 07_mareas_forzantes puede omitirse.")
    else:
        print("[OK] events.csv generado correctamente.")


def run_step(script_name: str, event: str, core_dir: Path) -> bool:
    """
    Ejecuta un script del core como subproceso:
      python <script> <event>
    Igual que run_event.py (sin flags extra).
    """
    script_path = core_dir / script_name
    if not script_path.exists():
        print(f"[ERROR] No existe el script: {script_path}")
        return False

    print("\n" + "=" * 70)
    print(f"  Ejecutando: {script_name}  (EVENTO = {event})")
    print("=" * 70)

    cmd = [sys.executable, str(script_path), event]
    print(f"  Comando: {' '.join(cmd)}")
    print("-" * 70)

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(core_dir))
    dt = time.time() - t0

    if res.returncode != 0:
        print(f"\n[ERROR] {script_name} terminó con código {res.returncode}")
        print("       Parando el pipeline para este evento.")
        return False

    print(f"\n[OK] {script_name} completado en {dt:.1f} s")
    return True


def detect_events(data_dir: Path) -> List[str]:
    """
    Devuelve todas las carpetas dentro de data/ como eventos,
    excepto las excluidas.
    """
    events: List[str] = []
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name in EXCLUDE_DIRS:
            continue
        if name.startswith("."):
            continue
        events.append(name)

    events.sort()
    return events


# =========================
# MAIN
# =========================
def main() -> None:
    core_dir = Path(__file__).resolve().parent
    project_root = project_root_from_core(core_dir)
    data_dir = project_root / "data"

    print("=" * 70)
    print("  PIPELINE TAMC 24h — MODO AUTOMÁTICO (data/*)")
    print(f"  Carpeta core: {core_dir}")
    print(f"  Carpeta data: {data_dir}")
    print("=" * 70)

    if not data_dir.exists():
        raise SystemExit(f"[ERROR] No existe {data_dir}")

    # Igual que run_event.py
    ensure_events_csv(project_root)

    events = detect_events(data_dir)

    print(f"\n[INFO] Directorios detectados en data/: {len(events)}")
    for ev in events:
        print("  -", ev)

    # Ejecutar pipeline para cada evento
    for event in events:
        print("\n" + "#" * 70)
        print(f"  EVENTO: {event}")
        print("#" * 70)

        start_all = time.time()

        for script in PIPELINE_STEPS:
            ok = run_step(script, event, core_dir)
            if not ok:
                print(f"[STOP] Pipeline detenido para {event}")
                break

        total_dt = time.time() - start_all
        print("\n" + "=" * 70)
        print(f"  Terminado EVENTO = {event}")
        print(f"  Tiempo total: {total_dt/60:.1f} min")
        print("=" * 70)

    print("\n[FIN] Pipeline completado para todos los directorios en data/.")


if __name__ == "__main__":
    main()

