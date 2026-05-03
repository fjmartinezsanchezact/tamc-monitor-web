#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_event.py

Lanza en cadena el pipeline TAMC 24h para:
- UN evento individual (que tenga raw/ o preprocessed/)
- O un "evento contenedor" que tenga subcarpetas:
    mainshock/
    control_*/

Orden:
  00_preprocess_24h.py
  01_rotation_3c.py
  02_tamc_metrics.py
  03_scan_criticality.py
  04_sync_multistation.py
  05_mareas_forzantes.py
  06_plotting.py
  07_robust_precursors_single.py

Uso:
  python run_event.py tohoku2011
  python run_event.py 2017_Tehuantepec_..._044919/mainshock
  python run_event.py 2017_Tehuantepec_..._044919   # corre mainshock + controles
"""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from typing import List


def project_root_from_core(core_dir: Path) -> Path:
    # core_dir = .../src/core
    return core_dir.parent.parent  # raíz del proyecto


def ensure_events_csv(project_root: Path) -> None:
    """
    Garantiza que src/events.csv exista.
    Si no existe, intenta generarlo llamando a src/build_events_csv.py.
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
        print("[WARN] No se pudo generar events.csv. (05_mareas_forzantes puede fallar si lo necesita)")
    else:
        print("[OK] events.csv generado correctamente.")


def ensure_preprocessed(event: str, project_root: Path, core_dir: Path) -> None:
    """
    Garantiza que exista:
      data/<event>/preprocessed/mseed_24h

    Si NO existe pero SÍ existe:
      data/<event>/raw

    entonces lanza automáticamente:
      python 00_preprocess_24h.py <event>
    """
    data_dir = project_root / "data" / Path(event)
    pre_dir = data_dir / "preprocessed" / "mseed_24h"
    raw_dir = data_dir / "raw"

    if pre_dir.exists():
        return

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] No existe preprocessed ni raw para '{event}'. "
            f"Esperaba al menos: {raw_dir}"
        )

    print("[INFO] No existe preprocessed/mseed_24h. Ejecutando 00_preprocess_24h.py automáticamente...")
    script_path = core_dir / "00_preprocess_24h.py"
    if not script_path.exists():
        raise FileNotFoundError(f"[ERROR] No existe el script: {script_path}")

    cmd = [sys.executable, str(script_path), event]
    print(f"  Comando: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(core_dir))

    if res.returncode != 0:
        raise RuntimeError("[ERROR] 00_preprocess_24h.py falló. No puedo continuar.")

    if not pre_dir.exists():
        raise RuntimeError(
            f"[ERROR] 00_preprocess_24h terminó pero NO creó: {pre_dir}\n"
            f"Revisa logs del preprocesado (quizá no encontró 3 componentes por estación)."
        )


def run_step(script_name: str, event: str, core_dir: Path) -> bool:
    """
    Ejecuta un script del core como subproceso:
      python <script> <event>
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
        print("       Parando el pipeline.")
        return False

    print(f"\n[OK] {script_name} completado en {dt:.1f} s")
    return True


def is_individual_event_dir(event_dir: Path) -> bool:
    """
    Determina si una carpeta de evento es "procesable" directamente:
      - tiene raw/
      - o tiene preprocessed/mseed_24h/
    """
    if (event_dir / "raw").exists():
        return True
    if (event_dir / "preprocessed" / "mseed_24h").exists():
        return True
    return False


def discover_sub_events(container_dir: Path) -> List[str]:
    """
    Detecta sub-eventos dentro de un contenedor:
      - mainshock
      - control_*
    que sean procesables (tienen raw/ o preprocessed/mseed_24h)
    """
    subs: List[str] = []
    if not container_dir.exists():
        return subs

    for d in sorted(container_dir.iterdir()):
        if not d.is_dir():
            continue

        if d.name == "mainshock" or d.name.startswith("control_"):
            if is_individual_event_dir(d):
                subs.append(d.name)

    return subs


def run_pipeline_for_event(event: str, project_root: Path, core_dir: Path) -> bool:
    """
    Ejecuta el pipeline completo para un evento individual.
    Devuelve True si terminó OK, False si falló.
    """
    print("=" * 70)
    print(f"  PIPELINE TAMC 24h — EVENTO: {event}")
    print(f"  Carpeta core: {core_dir}")
    print(f"  Project root: {project_root}")
    print("=" * 70)

    # Para mareas_forzantes (si lo requiere)
    ensure_events_csv(project_root)

    try:
        ensure_preprocessed(event, project_root, core_dir)
    except Exception as e:
        print(str(e))
        print("[ERROR] No se pudo asegurar preprocesado. Parando este evento.")
        return False

    steps: List[str] = [
        "00_preprocess_24h.py",
        "01_rotation_3c.py",
        "02_tamc_metrics.py",
        "03_scan_criticality.py",
        "04_sync_multistation.py",
        "05_mareas_forzantes.py",
        "06_plotting.py",
        "07_robust_precursors_single.py",
    ]

    start_all = time.time()
    for script in steps:
        ok = run_step(script, event, core_dir)
        if not ok:
            total_dt = time.time() - start_all
            print("\n" + "=" * 70)
            print(f"  Pipeline FALLÓ para EVENTO = {event}")
            print(f"  Tiempo hasta fallo: {total_dt/60:.1f} min")
            print("=" * 70)
            return False

    total_dt = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"  Pipeline terminado para EVENTO = {event}")
    print(f"  Tiempo total: {total_dt/60:.1f} min")
    print("=" * 70)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python run_event.py <EVENTO>")
        sys.exit(1)

    event = sys.argv[1].strip()

    core_dir = Path(__file__).resolve().parent
    project_root = project_root_from_core(core_dir)

    # ¿Es evento individual o contenedor?
    event_dir = project_root / "data" / Path(event)

    # Si es directamente procesable (tiene raw/ o preprocessed), corre como siempre.
    if is_individual_event_dir(event_dir):
        ok = run_pipeline_for_event(event, project_root, core_dir)
        sys.exit(0 if ok else 1)

    # Si no es procesable, intentamos tratarlo como contenedor.
    sub_events = discover_sub_events(event_dir)
    if not sub_events:
        print(f"[ERROR] '{event}' no parece ser un evento procesable ni un contenedor con mainshock/control_*.")
        print(f"        Revisá que exista: {event_dir}")
        print("        y que adentro haya subcarpetas mainshock/ o control_*/ con raw/ (o preprocessed/mseed_24h).")
        sys.exit(1)

    print("=" * 70)
    print(f"[INFO] Evento contenedor detectado: {event}")
    print("[INFO] Sub-eventos encontrados:")
    for name in sub_events:
        print(f"   - {name}")
    print("=" * 70)

    # Ejecutar todos (mainshock primero si existe)
    ordered = []
    if "mainshock" in sub_events:
        ordered.append("mainshock")
    ordered.extend([s for s in sub_events if s != "mainshock"])

    any_fail = False
    for sub in ordered:
        sub_event = f"{event}/{sub}"
        ok = run_pipeline_for_event(sub_event, project_root, core_dir)
        if not ok:
            any_fail = True
            # seguimos con los demás para que no se corte todo por uno
            # (si preferís que corte al primero que falla, decime y lo cambio)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
