# `src/core` — Pipeline TAMC Sísmico (00–22)

Este folder contiene el **pipeline principal** (scripts numerados) para:

- preprocesado 24h
- métricas TAMC
- scan de criticalidad por estación
- sincronía multi-estación
- precursors
- null tests (incluyendo el 09)
- comparación multi-evento + clustering
- tablas finales para paper (p-values + ELI)

> El runner recomendado para uso normal está en el root:  
> `run_full_pipeline_franjamarv2.py`

---

## Cómo se ejecuta (lo típico)

### Opción A — Runner interactivo (recomendado)

Desde el root del repo:

```bash
python run_full_pipeline_franjamarv2.py
```

### Opción B — Evento individual (modo automático)

Desde `src/core/`:

```bash
python run_event.py <EVENTO>
```

Este runner corre en cadena:

00 → 01 → 02 → 03 → 04 → 05 → 06 → 07  
(ver `run_event.py`).  

### Opción C — Todos los eventos en `data/`

Desde `src/core/`:

```bash
python run_event_datos.py
```

---

## Numeración de scripts (00–22)

### Core por evento (00–07)

- `00_preprocess_24h.py` — preprocesado 24h
- `01_rotation_3c.py` — rotación 3C (si aplica)
- `02_tamc_metrics.py` — métricas TAMC (CSV por estación + allstations)
- `03_scan_criticality.py` — scan de extremos / criticalidad
- `04_sync_multistation.py` — sincronía multi-estación
- `05_mareas_forzantes.py` — mareas/forzantes (si aplica)
- `06_plotting.py` — plots core
- `07_robust_precursors_single.py` — precursors por evento

---

### Null tests (08–12)

- `08_null_local_significance.py` — null local (por estación)
- `08_null_event_vs_controls.py` — null global (evento vs controles)  
  > Nota: hay dos scripts 08 (son variantes). El runner usa el que corresponda.
- `09_null_block_shuffle.py` — block-shuffle null (**sí se corre en v2 si existe**)
- `10_placebo_matched_controls_FINAL.py` — placebo con controles emparejados
- `11_null_event_vs_controls.py` — null evento vs controles (versión)
- `12_null_event_vs_controls_MULTI.py` — multi-control / multi-event

---

### Multi-evento + robustez + tablas (13–20)

- `13_compare_events_phases.py` — compara fases entre eventos
- `14_station_coverage_phases.py` — cobertura por fase
- `15_null_tests_13_14.py` — **script unificado**: REAL + Null A/B + Robustez C
- `15_null_viz_pvalues_13_14.py` — plots + p-values empíricos
- `16_make_pvalues_table_for_paper.py` — tabla LaTeX/CSV paper-ready
- `17_nullA_and_C_robustness.py` — robustez adicional (NullA + C)
- `18_extract_final_pvalues.py` — extrae p-values finales
- `19_cluster_events.py` — clustering de eventos
- `20_robustness_zthr_summary.py` — figura robustez vs zthr

---

### ELI + tablas finales (21–22)

- `21_explosion_index.py` — calcula el **Explosion-Likeness Index (ELI)**
- `22_make_eli_table_for_paper.py` — tablas LaTeX paper-ready para ELI

---

## Outputs (carpetas típicas)

- `resultados/<EVENTO>/...`  
  Outputs por evento (metrics, scan, sync, precursors, etc.)

- `resultados/null_tests_13_14/`  
  Outputs del bloque unificado (Tests 13–14 + nulls + robustez)

- `resultados/_summary/`  
  Outputs finales agregados (ELI, tablas para paper, etc.)

---

## Nota importante sobre el 09

`09_null_block_shuffle.py` **sí forma parte del pipeline**.

Si no te aparece en tu instalación:
- revisa que el archivo exista dentro de `src/core/`
- y que estés usando el runner `run_full_pipeline_franjamarv2.py`

Para plotear el resultado del 09 manualmente:
- usa `plot_block_shuffle_null.py`

