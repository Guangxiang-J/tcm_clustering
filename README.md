# TCM Constitution Clustering Analysis

A cleaned and Git-ready version of the three clustering analysis scripts:

- `main_analysis`: baseline comparison across 3 dimensionality reduction methods × 3 clustering methods.
- `main_stability`: repeated-run stability analysis with representative-seed selection and label alignment.
- `threshold_sensitivity`: sensitivity analysis across multiple binarization thresholds.

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── run_main_analysis.py
│   ├── run_main_stability.py
│   └── run_threshold_sensitivity.py
└── src/
    └── tcm_clustering/
        ├── __init__.py
        ├── common.py
        ├── main_analysis.py
        ├── main_stability.py
        └── threshold_sensitivity.py
```

## Installation

Create a clean environment and install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Expected data format

- Input file can be `.xlsx`, `.xls`, or `.csv`.
- Trait columns are excluded from clustering.
- By default, the trait columns are:

```text
YiDC YaDC QDC PDC DHC BSC SDC QSC
```

All remaining columns are treated as symptom variables.

## Usage

### 1) Main analysis

```bash
python scripts/run_main_analysis.py \
  --input data/clean_8_binary.xlsx \
  --output results/main_analysis
```

### 2) Main stability analysis

```bash
python scripts/run_main_stability.py \
  --input data/clean_8_binary.xlsx \
  --output results/main_stability_representative
```

### 3) Threshold sensitivity analysis

```bash
python scripts/run_threshold_sensitivity.py \
  --input data/clean_8_COPY.xlsx \
  --output results/threshold_sensitivity \
  --thresholds 2 3 4
```

## Notes on refactoring

Compared with the original scripts, this version:

1. extracts shared logic into `src/tcm_clustering/common.py`;
2. replaces top-of-file manual parameter editing with CLI arguments;
3. standardizes output metadata via `run_summary.json`;
4. keeps output filenames and analytical logic close to the originals;
5. makes the codebase easier to review, rerun, and version-control.

## Reproducibility

The default parameters preserve the intent of the original scripts, including:

- KMeans/GMM initialization settings
- K-search ranges
- UMAP settings
- seed lists
- representative-seed selection rules
- prevalence- and formula-based top-symptom outputs

## Suggested first commit message

```text
Refactor clustering analysis scripts into a Git-ready package with CLI entry points
```
