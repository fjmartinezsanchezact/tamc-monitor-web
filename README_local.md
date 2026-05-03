# TAMC-FRANJAMAR v3

## New in v3 (Cross-Domain Evaluation)

Version 3 extends the framework to a repeated engineered launch sequence 
(Starship IFT-1–IFT-8), providing a controlled cross-domain evaluation 
of structural regime organization under externally assigned mission outcomes.

---

## Overview

This repository contains a **reference implementation of the TAMC-FRANJAMAR framework**:  
a retrospective and reproducible pipeline designed to study **collective statistical behavior in multistation seismic networks** using an **event-centered representation**.

The framework is:

- ❌ Not predictive  
- ❌ Not real-time (core mode)  
- ❌ Not result-driven  

It is intended as a **methodological and exploratory tool** for understanding how **collective structure emerges at the network level**.

---

## What TAMC-FRANJAMAR does (in one sentence)

Given an event time and a set of stations, the pipeline produces **network-level diagnostics**
(extremes, synchrony, forcing proxies, null tests) that characterize the
**collective dynamical regime** of the event.

---

## Key idea

The relevant signal is **not amplitude at individual stations**, but the **emergence of structure across the network**.

The framework tests whether an event behaves like:

- **Deep / strongly coupled impulsive source**  
- **Surface-driven / weakly coupled phenomenon**

This is achieved without machine learning, using interpretable multistation observables.

---

## Environment setup (IMPORTANT)

### Recommended (Conda)

```
conda create -n tamc python=3.10 -y
conda activate tamc
pip install -r requirements.txt
```

### Alternative (pip only)

```
python -m venv tamc_env
tamc_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Notes

- Python **3.10 recommended**
- Fixed environment improves reproducibility
- You can freeze dependencies with:

```
pip freeze > environment_frozen.txt
```

---

## Repository structure

```
tamcsismico-SP/
├─ src/
├─ data/
├─ resultados/
├─ figures/
├─ run_full_pipeline_franjamarv2.py
├─ requirements.txt
└─ README.md
```

---

## Quickstart

```
pip install -r requirements.txt
python run_full_pipeline_franjamarv2.py
```

---

## Outputs

- Event diagnostics (multi-panel figures)
- Null tests and control comparisons
- Summary tables (CSV / LaTeX)

---

## Reproducibility

- Fixed parameters
- No event-specific tuning
- Compatible with public seismic data sources (FDSN)

---

## Recent application

A rapid-response application to the **2026 M7.4 Miyako (Japan) earthquake** demonstrates:

- Detection of network-coherent structure
- No prior alignment or tuning
- Emergent collective behavior across stations

---

## Disclaimer

TAMC-FRANJAMAR is a **diagnostic framework**.

It can detect statistically rare network-level structure, but:

- ❌ Does not predict earthquakes
- ❌ Does not provide early warning
