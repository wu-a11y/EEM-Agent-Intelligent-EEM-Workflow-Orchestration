# EEM Agent Integration

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

## Project Overview

This project delivers an end-to-end automated workflow for EEM (Excitation-Emission Matrix) fluorescence analysis. It integrates data preprocessing, PARAFAC component analysis, OpenFluor-based similarity matching, and AI-assisted report generation in one orchestrated pipeline.

The unified entry script is `code/0EEM_agent.py`, which runs four stages in order:
1. Rayleigh scattering removal (YOLO + interpolation/KNN strategy)
2. PARAFAC decomposition and split-half validation
3. OpenFluor database component comparison
4. AI report generation from metrics and literature context

---

## Project Structure

```text
EEM智能体整合/
├── code/
│   ├── 0EEM_agent.py
│   ├── yolo_rayleigh_removal.py
│   ├── pac_main.py
│   ├── database_comparison.py
│   ├── generate_ai_report.py
│   └── EEMs_toolkit.py
├── data/
│   ├── raw/
│   ├── process/
│   └── result/
├── docs/
│   ├── best.pt
│   ├── API.txt
│   ├── classification_results.xlsx
│   ├── component_analysis_metric_formulas.md
│   └── openflour_knowledge_base/
├── outputs/
├── picture/
├── README.md
└── requirements.txt
```

---

## Development Guide

### Project Architecture

This project is organized around a four-stage analysis pipeline:

1. **Scattering Removal Module** (`yolo_rayleigh_removal.py`):
  - Reads raw EEM Excel files from `data/raw/`.
  - Detects Rayleigh scattering regions using YOLO (`docs/best.pt`).
  - Applies KNN/interpolation-based reconstruction and saves cleaned EEM data.
  - Outputs processed EEM files to `data/process/eem/` and figures to `picture/KNN_KNN/`.

2. **PARAFAC and Validation Module** (`pac_main.py`):
  - Loads cleaned EEM tensors and performs PARAFAC decomposition.
  - Executes split-half validation and computes key metrics (Factor Similarity, Core Consistency, Explained Rate).
  - Saves intermediate arrays/NPZ files and component outputs to `data/process/` and `data/result/`.

3. **Database Comparison Module** (`database_comparison.py`):
  - Compares target PARAFAC loadings with OpenFluor reference components.
  - Calculates overlap-based Tucker congruence scores.
  - Writes ranked comparison results to `data/result/comparison_results/`.

4. **AI Report Module** (`generate_ai_report.py`):
  - Aggregates metric tables and comparison results.
  - Reads corresponding literature markdown from the knowledge base.
  - Calls DeepSeek/OpenAI-compatible API for component interpretation.
  - Generates final report: `outputs/fluorescence_analysis_report.md`.

### Process Logic

1. Clean raw EEM data and remove scattering noise.
2. Build and validate PARAFAC models.
3. Match extracted components to OpenFluor references.
4. Merge analytical metrics and literature interpretation into one final report.

The architecture follows a clear closed loop of **preprocess -> decompose -> compare -> interpret/report**.

---

## Features

* **Feature 1**: End-to-end orchestration from raw EEM input to final markdown report through one entry script.
* **Feature 2**: Built-in split-half PARAFAC validation with quantitative quality metrics.
* **Feature 3**: OpenFluor similarity matching and AI-assisted interpretation for component-level explainability.

---

## Installation and Configuration

### System Requirements

* Python 3.11
* Windows (validated)

### Installation Steps

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Core Dependencies

* `ultralytics` / `torch`: YOLO detection and inference
* `numpy` / `pandas` / `scipy`: numerical computing and data processing
* `tensorly` / `tlviz`: PARAFAC modeling and evaluation
* `scikit-learn`: KNN-based reconstruction utilities
* `openpyxl` / `xlsxwriter`: Excel read/write
* `openai`: AI report generation API client

---

## Usage

### Basic Usage

```powershell
python code/0EEM_agent.py
```

### Step-by-Step Execution

```powershell
python code/yolo_rayleigh_removal.py
python code/pac_main.py
python code/database_comparison.py
python code/generate_ai_report.py
```

### Key Inputs

* Raw EEM files: `data/raw/*.xlsx`
* YOLO model: `docs/best.pt`
* API key file: `docs/API.txt`

### Main Outputs

* Processed EEM data: `data/process/eem/`
* PARAFAC outputs: `data/result/`
* Comparison tables: `data/result/comparison_results/`
* Final report: `outputs/fluorescence_analysis_report.md`

---

## Function/Script Responsibilities

| Script | Responsibility |
| ------ | -------------- |
| `0EEM_agent.py` | Unified four-stage pipeline orchestration |
| `yolo_rayleigh_removal.py` | Scattering detection and EEM reconstruction |
| `pac_main.py` | PARAFAC modeling and split-half validation |
| `database_comparison.py` | OpenFluor similarity matching |
| `generate_ai_report.py` | AI-assisted markdown report generation |
| `EEMs_toolkit.py` | Core EEM/PARAFAC utility functions |

---

## Frequently Asked Questions

1. **Question 1**: Why does the pipeline report `script not found`?
  **Answer**: Ensure all stage script names in `code/0EEM_agent.py` match actual files under `code/`.

2. **Question 2**: Why does report generation fail at API stage?
  **Answer**: Verify `docs/API.txt` exists and contains a valid API key.

3. **Question 3**: Why is `2_comparison_results.xlsx` missing?
  **Answer**: Ensure Stage 1-3 completed successfully before running `generate_ai_report.py`.

4. **Question 4**: Why is the report not written to `outputs/`?
  **Answer**: Check folder write permissions and review runtime logs for API/network exceptions.

---

## Contact Information

* Project Maintainer: Yue Wang
* Contact Email: 642544234@qq.com

---
Last Updated: 2026-03-19
