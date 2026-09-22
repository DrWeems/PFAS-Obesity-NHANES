# PFAS Exposure and Obesity Risk in U.S. Adults: A Cross-Cycle NHANES Analysis

This repository contains the reproducible Python analysis code for the manuscript:

> **"PFAS Exposure and Obesity Across Two NHANES Cycles: Persistence, Temporal Change,
> and Racial Differences, 2017–2018 and 2021–2023"**
> *Journal of Food Technology and Health* (MDPI), 2026

---

## Repository Structure

```
├── README.md
├── LICENSE
├── Analysis.py                              # Full reproducible analysis script
└── data/
    └── README.md                            # Instructions for downloading NHANES data
```

---

## Requirements

**Python ≥ 3.8** with the following packages:

```
numpy
pandas
scipy
matplotlib
requests
statsmodels
```

Install all dependencies at once:

```bash
pip install numpy pandas scipy matplotlib requests statsmodels
```

---

## Data

This study uses publicly available data from the **National Health and Nutrition
Examination Survey (NHANES)**, administered by the National Center for Health
Statistics (NCHS), Centers for Disease Control and Prevention (CDC).

Raw data files are **not included** in this repository. The script will attempt
to download them automatically from the CDC/NCHS website at runtime. If automated
download is blocked in your environment, download the following eight files manually
and place them in the `data/` directory:

| File | Cycle | Content |
|---|---|---|
| `DEMO_J.XPT` | 2017–2018 | Demographics |
| `PFAS_J.XPT` | 2017–2018 | PFAS laboratory |
| `BMX_J.XPT` | 2017–2018 | Body measures |
| `SMQ_J.XPT` | 2017–2018 | Smoking |
| `DEMO_L.XPT` | 2021–2023 | Demographics |
| `PFAS_L.XPT` | 2021–2023 | PFAS laboratory |
| `BMX_L.XPT` | 2021–2023 | Body measures |
| `SMQ_L.XPT` | 2021–2023 | Smoking |

**Download links:**
- NHANES 2017–2018: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
- NHANES 2021–2023: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2021

> **Survey weight note:** This script uses `WTSB2YR` for the 2017–2018 J-cycle
> (from `PFAS_J`) and `WTSPF2YR` for the 2021–2023 L-cycle, consistent with the
> harmonized PFAS analytes used in this study.

---

## Usage

**Basic run (auto-downloads data if needed):**

```bash
python Analysis.py --data-dir data --output-dir results
```

**If data files are already downloaded:**

```bash
python Analysis.py --data-dir data --output-dir results --no-download
```

### Output

All outputs are saved to the specified `--output-dir`:

```
results/
├── figures/
│   ├── Figure1_Forest_Plot.*
│   ├── Figure2_Concentrations.*
│   ├── Figure3_Race_Stratified.*
│   ├── Figure4_Conceptual_Framework.*
│   ├── FigureS1_Distributions.*
│   ├── FigureS2_Obesity_Prevalence.*
│   ├── FigureS3_Dose_Response.*
│   ├── FigureS4_Correlations.*
│   ├── FigureS5_Scatter.*
│   └── FigureS6_DAG.*
└── tables/
    ├── Table1_Characteristics.csv
    ├── Table2_Adjusted_OR.csv
    ├── Table_S1_Detection_Frequencies.csv
    ├── Table_S2a_PFAS_Spearman_Correlations_J.csv
    ├── Table_S2b_PFAS_Spearman_Correlations_L.csv
    ├── Table_S3_Sensitivity_Analyses.csv
    ├── Quartile_Dose_Response_Models.csv
    ├── Race_PFAS_Interaction_Tests.csv
    ├── Validation_Against_Manuscript.csv
    └── run_summary.json
```

After each run, review `tables/Validation_Against_Manuscript.csv` to confirm
numerical results are within rounding tolerance of manuscript-reported values.

---

## Analysis Overview

- Adults aged ≥ 20 years in the NHANES PFAS subsamples (2017–2018 and 2021–2023)
- Obesity defined as BMI ≥ 30 kg/m²; pregnant participants excluded
- Natural-log transformed PFAS exposure in all regression models
- Complex-survey weighted logistic regression using Taylor linearization
  (PSU-within-stratum) with cycle-specific PFAS subsample weights
- Primary covariates: age, sex, race/ethnicity, education, poverty-income ratio, smoking
- Race-stratified domain models and White-vs-Black PFAS interaction tests
- Sensitivity analyses: severe obesity (BMI ≥ 35), women only, exclusion of
  values above the 99th percentile

---

## License

This code is released under the [MIT License](LICENSE).
© 2026 Alabama A&M University