# QCD: SU(3) Algebraic Correspondence and One-Loop Recalculation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official numerical pipeline and reproducible code accompanying the paper:

> **"An Effective Medium Field Theory Framework: Elastic Mode—SU(3) Algebraic Correspondence, Soft-Wall Confinement, and Retarded Negative Running"**

The code is designed to be highly transparent, reproducible, and relies almost entirely on the **Python Standard Library**, ensuring that reviewers and readers can verify the claims in the paper with minimal environment setup.

## 📂 Repository Structure

*   **`su3_full_pipeline.py`**: The core theoretical pipeline. It includes:
    *   **Part A-C**: Explicit SU(3) matrix machinery, Gell-Mann basis, and three SU(2) subgroup embeddings (rank-8 coverage).
    *   **Part B**: One-loop Feynman-parameter fluctuation integrals (\(\ln \det\)) from which the beta function coefficient \(b_0=9\) emerges.
    *   **Part D**: One-loop Renormalization Group (RG) running of \(\alpha_s(\mu)\) and the Landau scale \(\Lambda_{\text{QCD}} \approx 246\) MeV.
    *   **Part E**: Discrete-link Metropolis Monte Carlo fluctuation sampling for the two-link plaquette model.
*   **`standalone_measurements.py`**: The effective medium PDE pipeline. It includes:
    *   Measurement 1: Delayed backfill free bubble (retardation smoothing).
    *   Measurement 2: Sliding zero-force point (instantaneous geometry).
    *   Measurement 3: Relaxed-background zero-force curve.
    *   Measurement 4: Quark confinement pull (soft-wall pressure and string tension extraction).

## ⚙️ Requirements & Dependencies

The core logic of both scripts requires **only the Python Standard Library** (`math`, `random`, `csv`, `os`).
*   **Optional**: `matplotlib` (>=3.0) is required *only* if you want to generate the publication-quality figures. If it is not installed, the scripts will still generate all CSV data and simply print a warning.

```bash
# Optional: Install matplotlib to generate figures
pip install matplotlib
