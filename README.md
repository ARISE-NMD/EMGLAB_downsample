# Understanding Downsampling Effects on EMGLAB

This repository contains the fully reproducible pipeline used in the paper:

**“How does downsampling affect needle electromyography signals? A generalisable workflow for understanding downsampling effects on high-frequency time series.”**

The pipeline evaluates how multiple downsampling algorithms (e.g. LTTB, M4, MinMax) impact:
- Raw needle electromyography (nEMG) signals,
- Feature representations extracted from those signals,
- Feature extraction speedup,
- Classification performance for neuromuscular disorders.

All experiments are conducted on the **EMGLAB** dataset.

Four of the investigated downsamplers have been implemented in a library called [tsdownsample](https://github.com/predict-idlab/tsdownsample) by Van de Donckt et al [1].

---

## Overview

The pipeline is composed of five main stages:

1. **Preprocessing**: Loading of the EMGLAB .bin files and fragmentation of the recordings into 2-second fragments.
2. **Downsampling**: Application of six different downsampling techniques across 26 factors up to 1000x.
3. **Feature extraction**: Extraction of a large amount of features using tsfresh [2].
4. **Feature selection**: Dimensionality reduction using Boruta selection [3].
5. **Classification**: Performance evaluation for distinguishing between ALS, control and Myopathic fragments using a Random Forest.

---

## Reproducibility and runtime considerations

Due to the large number of feature extraction and downsampling configurations, full pipeline execution can be computationally expensive and time-consuming.

To facilitate reproducibility and result inspection without re-running the entire pipeline, the complete set of generated outputs for the EMGLAB dataset has been archived on **Zenodo**.

https://zenodo.org/10.5281/zenodo.18223778

---

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```


## Dataset
Download the original EMGLAB dataset from Kaggle:

https://www.kaggle.com/datasets/lydialoubar/emglab?resource=download

Ensure that:
- All .bin and .hea files are placed in a single flat directory
- no subdirectory structure is used

## Running the pipeline
The pipeline is controlled via `src/main.py`. At minimum, you must specify the path to the EMGLAB dataset.

### Basic Usage
```bash
    python src/main.py --data-dir /path/to/emglab
```

### Command line arguments
- `--data-dir`: Path to the raw EMGLAB dataset folder (required).
- `--output-dir`: Path to save processed results (defaults to `./results`)
- 
- `--fc-parameters`: `minimal`, `efficient` (default) or `comprehensive` (tsfresh settings).
- `--boruta-max-iter`: Number of maximum boruta iterations (default 100).
- `--cv-folds`: Number of cross validation folds (default 10).
- `--force`: Re-run all steps even if previous results exist.


## Output structure
Upon completion of the pipeline, the `--output-dir` will contain:
- `config.json`: a record of the parameter used
- `logs/`: logs for each pipeline stage
- `preprocessed/`, `downsampled/`, `features/`, `selected/`, `classification/` holding the data generated during the pipeline run.


## Applying the pipeline to other datasets
This pipeline is designed to be largly dataset agnostic beyond the preprocessing stage.

To apply the workflow to a different time series dataset, the primary changes are expected to be in the preprocessing step including:
- Data loading and parsing logic
- Signal segmentation
- Label extraction and segmenting

Once the data are transformed into the expected intermediate representation, the remaining stages can be used without modification.

## Notebooks
The `notebooks/` directory contains the Jupyter notebooks used to generate all figures presented in the paper. They are intended for transparancy and result reproduction and are not required to run the main pipeline itself.

## Citation
If you use this code or the associated datasets, please cite:

    Author1, Author2. (2026).
    How does downsampling affect needle electromyography signals? A generalisable workflow for understanding downsampling effects on high-frequency time series.
    Journal Name

Alternatively, use the “Cite this repository” button on GitHub to obtain a BibTeX entry.


## References

[1] Jeroen Van Der Donckt, Jonas Van Der Donckt, and Sofie Van Hoecke. tsdownsample:  High-performance time series downsampling for scalable visualization. SoftwareX, 29:102045, February 2025.

[2] Maximilian Christ, Nils Braun, Julius Neuffer, and Andreas W. Kempa-Liehr. Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307:72–77, September 2018.

[3] Miron B. Kursa, Aleksander Jankowski, and Witold R. Rudnicki. Boruta – A System for Feature Selection. Fundamenta Informaticae, 101(4):271–285, July 2010.