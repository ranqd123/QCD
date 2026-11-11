# QCD: SU(3) Algebraic Correspondence and One-Loop Recalculation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official numerical pipeline and reproducible code accompanying the paper:

> **"An Effective Medium Field Theory Framework: Elastic Mode—SU(3) Algebraic Correspondence, Soft-Wall Confinement, and Retarded Negative Running"**

The code is designed to be highly transparent and relies almost entirely on the **Python Standard Library**, ensuring that reviewers and readers can verify the claims in the paper with minimal environment setup.

## 📂 Repository Structure

*   **`su3_full_pipeline.py`**: The core theoretical and algebraic pipeline.
    *   **Part A-C**: Explicit SU(3) matrix machinery, Gell-Mann basis, and three SU(2) subgroup embeddings (rank-8 coverage).
    *   **Part B**: One-loop Feynman-parameter fluctuation integrals (\(\ln \det\)) from which the beta function coefficient \(b_0=9\) emerges.
    *   **Part D**: One-loop Renormalization Group (RG) running of \(\alpha_s(\mu)\) and the Landau scale \(\Lambda_{\text{QCD}} \approx 246\) MeV.
    *   **Part E**: Discrete-link Metropolis Monte Carlo fluctuation sampling for the two-link plaquette model.
*   **`standalone_measurements.py`**: The effective medium PDE and thermodynamic pipeline.
    *   **Measurement 1**: Delayed backfill free bubble (retardation smoothing and dynamic cavitation).
    *   **Measurement 2**: Sliding zero-force point (instantaneous geometry).
    *   **Measurement 3**: Relaxed-background zero-force curve.
    *   **Measurement 4**: Quark confinement pull (soft-wall pressure and string tension extraction).

## ⚙️ Requirements & Dependencies

The core logic of both scripts requires **only the Python Standard Library** (`math`, `random`, `csv`, `os`).
*   **Optional**: `matplotlib` (>=3.0) is required *only* if you want to generate the publication-quality figures. If it is not installed, the scripts will still generate all CSV data and print a warning.

```bash
# Optional: Install matplotlib to generate figures
pip install matplotlib
```

## 🚀 Quick Start

Clone the repository and run the scripts directly from your terminal. Both scripts are self-contained and run in under a minute on a standard laptop.

### 1. SU(3) Algebraic & One-Loop Pipeline
```bash
python su3_full_pipeline.py
```
**Outputs:**
*   `su3_pipeline_data/b0_emergence_data.csv`: Data showing \(b_0\) emerging from gluon, ghost, and quark integrals.
*   `su3_pipeline_data/rg_running_data.csv`: The \(\alpha_s(\mu)\) running curve.
*   `su3_pipeline_data/mc_plaquette_data.csv`: Metropolis Monte Carlo results.
*   **Terminal prints**: Anchor checks (e.g., \(b_0=9\), \(\Lambda \approx 246\) MeV, Haar limit \(8/9\)).

### 2. Effective Medium & Confinement Pipeline
```bash
python standalone_measurements.py
```
**Outputs:**
*   Four CSV files inside the `python_measurements/` folder.
*   Four high-resolution `.pdf` and `.png` figures (if `matplotlib` is installed).
*   **Terminal prints**: Anchor checks verifying the string tension \(F = -\sigma\) and the dynamic cavitation effect (peak near \(\tau/T \sim 1\)).

## 📊 Honest Scope

The simulations provided here are *effective minimal models* designed to reproduce the theoretical conclusions of the paper:
*   The \(SU(3)\) algebraic map and one-loop \(b_0\) calculation are **exact recalculations** based on standard background field methods.
*   The macroscopic confinement force and cavitation response are **phenomenological 1D PDE and Metropolis simulations**.
*   All anchor checks (`[PASS]`) printed in the terminal serve as internal consistency checks for the numerical implementation.

## 📖 Citation

If you use this code or the accompanying theoretical framework in your research, please cite our paper:

```bibtex
@article{YourLastName2026QCD,
  title = {An Effective Medium Field Theory Framework: Elastic Mode—SU(3) Algebraic Correspondence, Soft-Wall Confinement, and Retarded Negative Running},
  author = {Your Name},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year = {2026}
}
```

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
The code is provided "as is", without warranty of any kind, and is free for academic and commercial use.
