# Leukemia Survival Prediction

Prediction of overall survival in patients with myeloid leukemia, based on the [Data Challenge](https://challengedata.ens.fr/challenges/162) proposed by the ENS (École Normale Supérieure).

This project was carried out as part of a **Machine Learning course unit**. Its objective was to apply the full data science workflow, from exploratory data analysis to survival modeling and evaluation, on real clinical data.

> **Scope.** The challenge provides three datasets: a survival target, clinical data, and molecular (somatic mutation) data. **Due to time constraints, this project only processes the clinical dataset.** The molecular data is a natural extension for future work.

## Problem

The goal is to predict the risk of death of patients with blood cancer. Prognostic estimation is essential to adapt the therapeutic strategy: high-risk patients may be offered more intensive therapies, while low-risk patients may rather receive supportive care.

The target is the overall survival (OS) of each patient, described by two variables:

- `OS_YEARS`: follow-up duration in years since diagnosis
- `OS_STATUS`: status at last follow-up (0: alive, 1: deceased)

The data is **right-censored**: for patients still alive at the end of the follow-up, the death event has not been observed and can only be said to occur after the last follow-up. Any valid analysis and model must account for this censoring.

## Data

The clinical dataset covers **3,323 patients** followed across **23 clinical centers**, with:


| Variable       | Description                                            |
| -------------- | ------------------------------------------------------ |
| `CENTER`       | Clinical center where the patient is treated           |
| `BM_BLAST`     | Proportion of abnormal blood cells in the bone marrow  |
| `WBC`          | White blood cell count (Giga/L)                           |
| `ANC`          | Absolute neutrophil count (Giga/L)                        |
| `MONOCYTES`    | Monocyte count (Giga/L)                                   |
| `HB`           | Hemoglobin level (g/dL)                                |
| `PLT`          | Platelet count (Giga/L)                                   |
| `CYTOGENETICS` | Karyotype in ISCN notation (chromosomal abnormalities) |


Key findings of the exploratory analysis:

- **49.6% censoring rate** (1,600 censored patients out of 3,173 with a valid target); 150 patients have a missing target and are excluded.
- Deaths are concentrated in the first years after diagnosis (median follow-up of 1.38 years for deceased patients), suggesting strong early lethality.
- Hematological parameters are right-skewed (except `HB`), motivating a `log1p` transformation for the Cox model.
- Missing values are **not missing at random**: 18.1% for `MONOCYTES` (not collected at all by two centers), 11.6% for `CYTOGENETICS`, 8.2% for `WBC`, and strongly center-dependent.

## Methodology

### Preprocessing and feature engineering

- Median imputation of missing blood parameters, fitted on the training set only (no leakage).
- `log1p` transformation of skewed blood parameters.
- Categorical `CENTER` encoded with one-hot encoding.
- **Cytogenetics risk score**: the raw karyotype (1,194 unique values) is mapped to the prognostic risk groups of the ELN genetic classification (very good, good, intermediate, poor, very poor), based on [Grimwade &amp; Hills, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4874200/). Abnormality counting uses a simple proxy (comma counting), tolerant to ISCN notation variants.

### Evaluation protocol

- **Metric**: IPCW C-index (inverse probability of censoring weighted), computed at a 7-year horizon. Unlike the standard C-index, it accounts for right censoring when weighting comparable patient pairs.
- **Cross-validation**: 5-fold `StratifiedKFold`, stratified on `OS_STATUS` to keep the death/censoring ratio constant across folds.
- Model improvements are validated with **paired Student t-tests** on the fold scores.

## Results

IPCW C-index at 7 years, 5-fold cross-validation (mean ± std):


| Model                              | C-index             | Notes                                           |
| ---------------------------------- | ------------------- | ----------------------------------------------- |
| BM\_BLAST only                     | 0.6156 ± 0.0088     | Clinical baseline                               |
| LightGBM regressor                 | 0.5997 ± 0.0220     | Predicts `OS_YEARS`, ignores censoring          |
| Cox PH                             | 0.6847 ± 0.0278     | Significantly better than LightGBM (p = 0.0015) |
| Cox PH + cytogenetic risk groups   | 0.6907 ± 0.0259     |                                                 |
| Cox PH, without missing indicators | 0.6912 ± 0.0261     | Simpler model, kept                             |
| Coxnet (L1-penalized Cox)          | **0.6952 ± 0.0295** | Final model                                     |


The final Coxnet model, trained on the training split, reaches an **IPCW C-index of 0.686** on the held-out test set.

Key learnings:

- A standard regression model (LightGBM) that ignores censoring performs no better than the single `BM_BLAST` feature: properly accounting for the survival structure of the target is what drives the performance.
- The learned coefficients are clinically consistent: `CYTO_LVL_VERY_POOR`, high `BM_BLAST` and some centers increase the predicted risk, while `HB`, `CYTO_LVL_NORMAL` and other centers decrease it.

## Repository structure

```
.
├── notebooks
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02a_preprocessing.ipynb       # Missing values, log transforms, joins
│   ├── 02b_feature_engineering.ipynb # Cytogenetics risk groups
│   ├── 03_evaluation_protocol.ipynb  # CV strategy and IPCW C-index
│   ├── 04_survival_modeling.ipynb    # Cox, Coxnet and baselines
│   └── restitution.ipynb             # Summary notebook (course restitution)
├── src/leukemia
│   ├── data                          # Dataset loading and joining
│   ├── features                      # Feature engineering (cytogenetics)
│   └── models                        # Evaluation protocol and metrics
├── environment.yml
├── pyproject.toml
└── LICENSE
```

## Installation

Requires Python 3 with `pandas`, `numpy`, `scikit-learn`, `scikit-survival` and `lightgbm`.

```bash
# Conda environment (recommended)
conda env create -f environment.yml
conda activate leukemia_survival_prediction_env

# Install the leukemia package
pip install -e .
```

## Usage

1. Download the data from the [challenge page](https://challengedata.ens.fr/challenges/162) (the data is not included in this repository).
2. Place the files according to the paths expected by the notebooks (see `notebooks/01_eda.ipynb`).
3. Run the notebooks in numerical order: they reproduce the full pipeline, from EDA to the final Coxnet model and its evaluation.

## References

- [Challenge, Data Challenge ENS](https://challengedata.ens.fr/challenges/162)
- [Grimwade D. &amp; Hills R.K., Independent prognostic factors for AML outcome, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4874200/)
- [scikit-survival documentation](https://scikit-survival.readthedocs.io/)

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
