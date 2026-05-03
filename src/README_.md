# src/core/

This directory contains the **core modules of the TAMC-FRANJAMAR pipeline**.

Each file implements a **single, well-defined stage** of the analysis.  
Modules are executed sequentially or selectively by the main runner (`run_full_pipeline_franjamarv2.py`).

The numbering (`00–22`) reflects the **logical execution order**, not independent scripts.

---

## Design philosophy

- **Modular but not ad hoc**  
  Each module has a fixed role within the pipeline.

- **Fixed-parameter execution**  
  No module performs event-specific tuning or adaptive optimization.

- **System-level focus**  
  Individual modules operate on station-level signals but produce outputs intended for collective interpretation.

- **Failure is allowed**  
  Modules are designed to produce null or weak results without interrupting the pipeline.

---

## Module overview

### Preprocessing and feature extraction
- `00_preprocess_24h.py` — waveform preprocessing and window extraction  
- `01_rotation_3c.py` — optional 3-component rotation  
- `02_tamc_metrics.py` — station-level metric computation  

### Detection and aggregation
- `03_scan_criticality.py` — detection of statistically extreme activity  
- `04_sync_multistation.py` — multistation synchronization metrics  
- `05_mareas_forzantes.py` — external forcing proxies (e.g., tides)  

### Visualization
- `06_plotting.py` — standardized plotting routines  

### Robustness and validation
- `07_robust_precursors_single.py` — single-event robustness checks  
- `08_null_event_vs_controls.py` — event vs control null tests  
- `08_null_local_significance.py` — local significance evaluation  
- `09_null_block_shuffle.py` — block-shuffle null models  
- `10_placebo_matched_controls_FINAL.py` — placebo tests  

### Inter-event analysis
- `11_null_event_vs_controls.py` — alternative null contrasts  
- `12_null_event_vs_controls_MULTI.py` — multi-event null analysis  
- `13_compare_events_phases.py` — inter-event phase comparison  
- `14_station_coverage_phases.py` — network coverage diagnostics  
- `15_null_tests_13_14.py` — null validation for inter-event metrics  
- `15_null_viz_pvalues_13_14.py` — visualization of null p-values  

### Synthesis and classification
- `16_make_pvalues_table_for_paper.py` — summary tables  
- `17_nullA_and_C_robustness.py` — robustness across null classes  
- `18_extract_final_pvalues.py` — final p-value extraction  
- `19_cluster_events.py` — event clustering and regime exploration  
- `20_robustness_zthr_summary.py` — threshold robustness summaries  
- `21_explosion_index.py` — computes the Explosion-Likeness Index (ELI) from TAMC outputs and ranks events  
- `22_make_eli_table_for_paper.py` — exports ELI rankings to LaTeX/CSV tables for the paper
  

---

## Execution

These modules are **not intended to be run manually**.

Execution is orchestrated by:
```bash
python run_full_pipeline_franjamarv2.py
```

Optionally, once the full pipeline has generated per-event summaries, you can compute the
**Explosion-Likeness Index (ELI)** and export paper-ready tables:
```bash
python 21_explosion_index.py
python 22_make_eli_table_for_paper.py
```


which ensures:
- consistent parameter passing  
- correct ordering  
- proper handling of missing or null results  

---

## Notes for users

- The default analysis window is **24h** for most events.
- For compact impulsive explosions (e.g., DPRK 2017) the pipeline may run in a **12h** configuration to avoid diluting the \(t=0\) packet.

- Outputs produced by these modules are written to `data/` and `resultados/`
- All outputs are machine-generated and reproducible
- Users are encouraged to inspect the code to understand methodological choices

---

This directory implements **how the framework works**,  
not **what the framework claims**.
