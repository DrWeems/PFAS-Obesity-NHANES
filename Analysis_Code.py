#!/usr/bin/env python3
"""
Supplementary File S1. Reproducible Python analysis for:

    PFAS Exposure and Obesity Across Two NHANES Cycles:
    Persistence, Temporal Change, and Racial Differences, 2017-2018 and 2021-2023

This script was rebuilt to implement the analysis described in the current manuscript:
  * Adults age >=20 years in the NHANES PFAS subsamples
  * Positive PFAS subsample weights and measured BMI
  * Pregnant participants excluded
  * Obesity = BMI >=30 kg/m^2
  * Natural-log PFAS exposure scale for regression models
  * Primary adjustment: age, sex, race/ethnicity, education, PIR, smoking
  * Cycle-specific complex-survey analysis using strata, PSU, and PFAS weights
  * Race-stratified domain models and White-vs-Black PFAS interaction tests
  * Sensitivity analyses: severe obesity, women only, >99th percentile exclusion
  * Crude models, weighted geometric means, weighted correlations, quartile models
  * Main-manuscript Figures 1-4 and Supplementary Figures S1-S6
  * CSV tables corresponding to manuscript and supplementary tables
  * Automated validation against numerical values printed in the manuscript

IMPORTANT WEIGHT NOTE
---------------------
The current manuscript text says WTSSBJ2Y for the 2017-2018 J-cycle. That variable
belongs to SSPFAS_J (the surplus-serum PFAS file), not PFAS_J. The harmonized PFAS
analytes used in this study are from PFAS_J, whose official subsample weight is
WTSB2YR. Therefore this reproducibility script uses WTSB2YR for J-cycle and
WTSPF2YR for L-cycle. Update the Methods sentence in the manuscript accordingly.

DATA
----
Expected NHANES public-use files:
    DEMO_J.XPT, PFAS_J.XPT, BMX_J.XPT, SMQ_J.XPT
    DEMO_L.XPT, PFAS_L.XPT, BMX_L.XPT, SMQ_L.XPT

If files are not present in --data-dir, the script attempts to download them from
CDC/NCHS. If CDC blocks automated download in your environment, download the eight
files manually and place them in --data-dir.

Example:
    python Supplementary_File_S1_Analysis_Code.py --data-dir data --output-dir results

Dependencies:
    numpy, pandas, scipy, matplotlib, requests, statsmodels

The survey estimators use Taylor linearization with PSU-within-stratum aggregation.
Domain analyses retain the full positive-weight PFAS design and assign zero score
contributions outside the analytic domain, preserving the NHANES survey structure.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:  # pragma: no cover
    lowess = None


# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------

CYCLE_LABELS = {"J": "2017-2018", "L": "2021-2023"}
WEIGHT_COL = {"J": "WTSB2YR", "L": "WTSPF2YR"}

RACE_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "NH White",
    4: "NH Black",
    6: "NH Asian",
    7: "Other/Multi",
}
RACE_ORDER = [
    "NH White",
    "NH Black",
    "Mexican American",
    "Other Hispanic",
    "NH Asian",
    "Other/Multi",
]

EDUC_ORDER = ["Less than HS", "HS/GED", "Some college", "College+"]

# Models exactly as shown in the main manuscript adjusted-OR table.
MODEL_ANALYTES: List[Tuple[str, str]] = [
    ("LBXNFOS", "n-PFOS"),
    ("LBXMFOS", "Sm-PFOS"),
    ("Total_PFOS", "Total PFOS"),
    ("LBXNFOA", "n-PFOA"),
    ("Total_PFOA", "Total PFOA"),
    ("LBXPFHS", "PFHxS"),
    ("LBXPFNA", "PFNA"),
    ("LBXPFDE", "PFDeA"),
    ("LBXMPAH", "MePFOSA-AcOH"),
    ("LBXPFUA", "PFUA"),
]

# Individual laboratory analytes shown in the manuscript geometric-mean table.
GM_ANALYTES: List[Tuple[str, str]] = [
    ("LBXNFOS", "n-PFOS"),
    ("LBXMFOS", "Sm-PFOS"),
    ("LBXNFOA", "n-PFOA"),
    ("LBXBFOA", "Sb-PFOA"),
    ("LBXPFHS", "PFHxS"),
    ("LBXPFNA", "PFNA"),
    ("LBXPFDE", "PFDeA"),
    ("LBXPFUA", "PFUA"),
    ("LBXMPAH", "MePFOSA-AcOH"),
]

# Table S2 / Supplementary Figure S4.
CORR_ANALYTES: List[Tuple[str, str]] = [
    ("LBXNFOS", "n-PFOS"),
    ("LBXMFOS", "Sm-PFOS"),
    ("LBXNFOA", "n-PFOA"),
    ("LBXPFHS", "PFHxS"),
    ("LBXPFNA", "PFNA"),
    ("LBXPFDE", "PFDeA"),
    ("LBXPFUA", "PFUA"),
    ("LBXMPAH", "MePFOSA-AcOH"),
]

DETECTION_MAP: List[Tuple[str, str, str]] = [
    ("LBXNFOS", "LBDNFOSL", "n-PFOS"),
    ("LBXMFOS", "LBDMFOSL", "Sm-PFOS"),
    ("LBXNFOA", "LBDNFOAL", "n-PFOA"),
    ("LBXBFOA", "LBDBFOAL", "Sb-PFOA"),
    ("LBXPFHS", "LBDPFHSL", "PFHxS"),
    ("LBXPFNA", "LBDPFNAL", "PFNA"),
    ("LBXPFDE", "LBDPFDEL", "PFDeA"),
    ("LBXPFUA", "LBDPFUAL", "PFUA"),
    ("LBXMPAH", "LBDMPAHL", "MePFOSA-AcOH"),
]

DISTRIBUTION_ANALYTES = ["LBXNFOS", "LBXNFOA", "LBXPFHS", "LBXPFNA", "LBXPFDE", "LBXPFUA"]
DOSE_RESPONSE_ANALYTES = ["LBXPFDE", "LBXPFHS", "LBXPFNA", "LBXNFOA", "LBXNFOS"]
SCATTER_ANALYTES = ["LBXNFOA", "LBXPFNA", "LBXPFDE", "LBXPFUA"]
RACE_FIG_ANALYTES = ["LBXNFOA", "Total_PFOA", "LBXPFNA"]

LABEL_BY_VAR = {v: lab for v, lab in (MODEL_ANALYTES + GM_ANALYTES + CORR_ANALYTES)}
LABEL_BY_VAR.update({"Total_PFOS": "Total PFOS", "Total_PFOA": "Total PFOA"})

# Official NHANES LLOD for these PFAS in both cycles.
LOD = 0.1

FILE_NAMES = {
    "J": {"DEMO": "DEMO_J.XPT", "PFAS": "PFAS_J.XPT", "BMX": "BMX_J.XPT", "SMQ": "SMQ_J.XPT"},
    "L": {"DEMO": "DEMO_L.XPT", "PFAS": "PFAS_L.XPT", "BMX": "BMX_L.XPT", "SMQ": "SMQ_L.XPT"},
}

# Multiple URL patterns are tried because NCHS has used both legacy and public paths.
DOWNLOAD_CANDIDATES = {
    "J": {
        key: [
            f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}",
            f"https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{name}",
        ]
        for key, name in FILE_NAMES["J"].items()
    },
    "L": {
        key: [
            f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{name}",
            f"https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/{name}",
        ]
        for key, name in FILE_NAMES["L"].items()
    },
}

# Main-manuscript numerical targets used only for validation, never for estimation.
EXPECTED_ADJUSTED_OR = {
    "J": {
        "n-PFOS": (0.774, 0.553, 1.083, 0.082),
        "Sm-PFOS": (0.825, 0.544, 1.250, 0.185),
        "Total PFOS": (0.780, 0.553, 1.100, 0.090),
        "n-PFOA": (0.813, 0.432, 1.531, 0.295),
        "Total PFOA": (0.797, 0.394, 1.614, 0.301),
        "PFHxS": (0.779, 0.478, 1.269, 0.159),
        "PFNA": (0.924, 0.500, 1.707, 0.635),
        "PFDeA": (0.756, 0.383, 1.492, 0.219),
        "MePFOSA-AcOH": (1.016, 0.479, 2.153, 0.937),
        "PFUA": (0.705, 0.377, 1.319, 0.138),
    },
    "L": {
        "n-PFOS": (0.753, 0.515, 1.100, 0.084),
        "Sm-PFOS": (0.885, 0.642, 1.221, 0.244),
        "Total PFOS": (0.772, 0.537, 1.109, 0.092),
        "n-PFOA": (0.792, 0.541, 1.160, 0.119),
        "Total PFOA": (0.765, 0.526, 1.110, 0.090),
        "PFHxS": (0.838, 0.592, 1.187, 0.160),
        "PFNA": (0.747, 0.571, 0.977, 0.043),
        "PFDeA": (0.570, 0.395, 0.823, 0.022),
        "MePFOSA-AcOH": (0.783, 0.461, 1.332, 0.186),
        "PFUA": (0.498, 0.264, 0.938, 0.042),
    },
}

EXPECTED_GM = {
    "J": {
        "n-PFOS": (3.232, 2.983, 3.502),
        "Sm-PFOS": (1.366, 1.246, 1.498),
        "n-PFOA": (1.396, 1.301, 1.498),
        "Sb-PFOA": (0.075, 0.072, 0.079),
        "PFHxS": (1.158, 1.067, 1.256),
        "PFNA": (0.442, 0.391, 0.499),
        "PFDeA": (0.202, 0.186, 0.220),
        "PFUA": (0.131, 0.120, 0.144),
        "MePFOSA-AcOH": (0.130, 0.121, 0.141),
    },
    "L": {
        "n-PFOS": (1.843, 1.709, 1.986),
        "Sm-PFOS": (0.754, 0.699, 0.814),
        "n-PFOA": (0.939, 0.843, 1.047),
        "Sb-PFOA": (0.079, 0.072, 0.087),
        "PFHxS": (0.799, 0.713, 0.896),
        "PFNA": (0.297, 0.274, 0.321),
        "PFDeA": (0.106, 0.101, 0.112),
        "PFUA": (0.089, 0.084, 0.095),
        "MePFOSA-AcOH": (0.087, 0.085, 0.089),
    },
}

EXPECTED_N = {"J": {"analytic": 1376, "complete": 1201}, "L": {"analytic": 2145, "complete": 1869}}
EXPECTED_INTERACTION_P = {"n-PFOA": 0.024, "Total PFOA": 0.040}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_xpt(path: Path) -> pd.DataFrame:
    """Read SAS transport file, tolerating common encodings."""
    try:
        return pd.read_sas(path, format="xport", encoding="latin1")
    except Exception:
        return pd.read_sas(path, format="xport")


def download_file(urls: Sequence[str], dest: Path, timeout: int = 90) -> bool:
    if requests is None:
        return False
    headers = {"User-Agent": "Mozilla/5.0 NHANES-reproducibility-script/1.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if r.ok and len(r.content) > 1024:
                dest.write_bytes(r.content)
                print(f"Downloaded {dest.name} from {url}")
                return True
        except Exception as exc:
            print(f"Download attempt failed for {url}: {exc}", file=sys.stderr)
    return False


def ensure_data_files(data_dir: Path, allow_download: bool = True) -> Dict[str, Dict[str, Path]]:
    paths: Dict[str, Dict[str, Path]] = {"J": {}, "L": {}}
    missing = []
    for cycle in ["J", "L"]:
        for component, filename in FILE_NAMES[cycle].items():
            # Accept either .XPT or .xpt and a few common duplicate-name variants.
            candidates = [
                data_dir / filename,
                data_dir / filename.lower(),
                data_dir / filename.replace(".XPT", ".xpt"),
            ]
            existing = next((p for p in candidates if p.exists()), None)
            if existing is None and allow_download:
                dest = data_dir / filename
                ensure_dir(data_dir)
                if download_file(DOWNLOAD_CANDIDATES[cycle][component], dest):
                    existing = dest
            if existing is None:
                missing.append(filename)
            else:
                paths[cycle][component] = existing
    if missing:
        raise FileNotFoundError(
            "Missing required NHANES files: " + ", ".join(missing) +
            ". Place them in --data-dir or rerun with internet access so the script can download them."
        )
    return paths


def merge_cycle(files: Mapping[str, Path]) -> pd.DataFrame:
    d = read_xpt(files["DEMO"]).copy()
    for component in ["PFAS", "BMX", "SMQ"]:
        d = d.merge(read_xpt(files[component]), on="SEQN", how="left", validate="one_to_one")
    return d


def clean_binary(series: pd.Series, yes=1, no=2) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[series.eq(yes)] = 1.0
    out.loc[series.eq(no)] = 0.0
    return out


def prep_cycle(d: pd.DataFrame, cycle: str) -> pd.DataFrame:
    d = d.copy()
    wtcol = WEIGHT_COL[cycle]
    if wtcol not in d.columns:
        raise KeyError(f"{wtcol} not found in {cycle}-cycle PFAS file.")

    # pandas can decode SAS transport numeric zero as a tiny positive floating value.
    # Genuine positive PFAS weights are many thousands, so >1 safely separates them.
    d["PFAS_WT"] = d[wtcol].where(d[wtcol] > 1, 0.0)
    d["cycle"] = cycle
    d["cycle_label"] = CYCLE_LABELS[cycle]

    d["pregnant"] = d.get("RIDEXPRG", pd.Series(np.nan, index=d.index)).eq(1)
    d["obese"] = np.where(d["BMXBMI"].notna(), (d["BMXBMI"] >= 30).astype(float), np.nan)
    d["severe_obese"] = np.where(d["BMXBMI"].notna(), (d["BMXBMI"] >= 35).astype(float), np.nan)
    d["female"] = clean_binary(d["RIAGENDR"], yes=2, no=1)
    d["ever_smoker"] = clean_binary(d["SMQ020"], yes=1, no=2)

    d["race"] = d["RIDRETH3"].map(RACE_MAP)
    d["education"] = pd.Series(np.nan, index=d.index, dtype=object)
    d.loc[d["DMDEDUC2"].isin([1, 2]), "education"] = "Less than HS"
    d.loc[d["DMDEDUC2"].eq(3), "education"] = "HS/GED"
    d.loc[d["DMDEDUC2"].eq(4), "education"] = "Some college"
    d.loc[d["DMDEDUC2"].eq(5), "education"] = "College+"

    d["Total_PFOS"] = d["LBXNFOS"] + d["LBXMFOS"]
    d["Total_PFOA"] = d["LBXNFOA"] + d["LBXBFOA"]

    # Primary analytic domain from manuscript.
    d["analytic_domain"] = (
        (d["PFAS_WT"] > 0)
        & (d["RIDAGEYR"] >= 20)
        & d["BMXBMI"].notna()
        & d["LBXNFOS"].gt(0)  # valid harmonized PFAS panel / n-PFOS result
        & (~d["pregnant"])
    )

    # A common complete-case domain used for Table 1 and sample-flow reporting.
    d["primary_complete"] = (
        d["analytic_domain"]
        & d["RIDAGEYR"].notna()
        & d["female"].notna()
        & d["race"].notna()
        & d["education"].notna()
        & d["INDFMPIR"].notna()
        & d["ever_smoker"].notna()
    )
    return d


# -----------------------------------------------------------------------------
# Complex survey helpers: Taylor linearization
# -----------------------------------------------------------------------------

def design_df(data: pd.DataFrame) -> int:
    des = data.loc[data["PFAS_WT"] > 0, ["SDMVSTRA", "SDMVPSU"]].dropna().drop_duplicates()
    n_psu = len(des)
    n_strata = des["SDMVSTRA"].nunique()
    return max(1, int(n_psu - n_strata))


def stratified_cluster_variance(data: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Variance of sums of linearized contributions by PSU within stratum."""
    cols = list(cols)
    base = data.loc[data["PFAS_WT"] > 0, ["SDMVSTRA", "SDMVPSU"] + cols].copy()
    psus = base[["SDMVSTRA", "SDMVPSU"]].dropna().drop_duplicates()
    agg = base.groupby(["SDMVSTRA", "SDMVPSU"], as_index=False)[cols].sum()
    agg = psus.merge(agg, on=["SDMVSTRA", "SDMVPSU"], how="left").fillna({c: 0.0 for c in cols})
    V = np.zeros((len(cols), len(cols)), dtype=float)
    lonely = 0
    for _, g in agg.groupby("SDMVSTRA"):
        Z = g[cols].to_numpy(float)
        m = len(Z)
        if m < 2:
            # Full NHANES designs normally have >=2 PSUs/stratum. For an unexpected
            # singleton in the source design, use a conservative centered contribution.
            lonely += 1
            z = Z[0][:, None]
            V += z @ z.T
            continue
        C = Z - Z.mean(axis=0)
        V += (m / (m - 1.0)) * (C.T @ C)
    if lonely:
        warnings.warn(f"Encountered {lonely} singleton strata in the positive-weight PFAS design.")
    return V


def survey_mean(data: pd.DataFrame, y: str, domain: pd.Series) -> Dict[str, float]:
    des = data.loc[data["PFAS_WT"] > 0].copy()
    I = domain.reindex(des.index).fillna(False).to_numpy(bool) & des[y].notna().to_numpy()
    if I.sum() == 0:
        return dict(estimate=np.nan, se=np.nan, lower=np.nan, upper=np.nan, var=np.nan, df=design_df(des), n=0)
    w = des["PFAS_WT"].to_numpy(float)
    yy = des[y].to_numpy(float)
    D = np.sum(w[I])
    mu = np.sum(w[I] * yy[I]) / D
    c = np.zeros(len(des), dtype=float)
    c[I] = w[I] * (yy[I] - mu) / D
    des["_lin"] = c
    var = float(stratified_cluster_variance(des, ["_lin"])[0, 0])
    se = math.sqrt(max(var, 0.0))
    df = design_df(des)
    crit = t.ppf(0.975, df)
    return dict(estimate=mu, se=se, lower=mu - crit * se, upper=mu + crit * se, var=var, df=df, n=int(I.sum()))


def survey_prop(data: pd.DataFrame, y: str, domain: pd.Series) -> Dict[str, float]:
    r = survey_mean(data, y, domain)
    p = r["estimate"]
    se = r["se"]
    if np.isfinite(p) and 0 < p < 1 and np.isfinite(se) and se > 0:
        crit = t.ppf(0.975, r["df"])
        logit_p = math.log(p / (1 - p))
        se_logit = se / (p * (1 - p))
        lo = 1 / (1 + math.exp(-(logit_p - crit * se_logit)))
        hi = 1 / (1 + math.exp(-(logit_p + crit * se_logit)))
    else:
        lo, hi = r["lower"], r["upper"]
    r.update(lower=max(0.0, lo) if np.isfinite(lo) else np.nan,
             upper=min(1.0, hi) if np.isfinite(hi) else np.nan)
    return r


def survey_gm(data: pd.DataFrame, y: str, domain: pd.Series) -> Dict[str, float]:
    d = data.copy()
    d["_ln_y"] = np.where(d[y] > 0, np.log(d[y]), np.nan)
    r = survey_mean(d, "_ln_y", domain)
    return dict(
        estimate=math.exp(r["estimate"]) if np.isfinite(r["estimate"]) else np.nan,
        lower=math.exp(r["lower"]) if np.isfinite(r["lower"]) else np.nan,
        upper=math.exp(r["upper"]) if np.isfinite(r["upper"]) else np.nan,
        se_log=r["se"], var_log=r["var"], df=r["df"], n=r["n"],
    )


def weighted_quantile(values: pd.Series, quantiles: Sequence[float], weights: pd.Series) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(quantiles, cdf, v)


def weighted_quartile_codes(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Assign approximately equal survey-weight quartiles, robust to tied LOD fill values.

    NHANES PFAS can have many identical LOD/sqrt(2) values, which can make ordinary
    cutpoint-based qcut/pd.cut fail because quartile boundaries are duplicated. This
    function orders observations by concentration and cumulative survey weight and
    assigns Q1-Q4 directly. Ties at a boundary may be split, matching the practical
    ntile-style behavior used to preserve four exposure categories for exploratory
    dose-response plots.
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    m = values.notna() & weights.notna() & (weights > 0)
    if not m.any():
        return out
    dd = pd.DataFrame({"v": values[m].astype(float), "w": weights[m].astype(float)})
    dd = dd.sort_values(["v"], kind="mergesort")
    mid = (dd["w"].cumsum() - 0.5 * dd["w"]) / dd["w"].sum()
    q = np.minimum(4, np.floor(mid.to_numpy() * 4).astype(int) + 1)
    out.loc[dd.index] = q
    return out


def covariate_frame(data: pd.DataFrame, include_race: bool = True, include_sex: bool = True) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x["RIDAGEYR"] = data["RIDAGEYR"]
    if include_sex:
        x["female"] = data["female"]
    if include_race:
        # NH White is reference.
        for lab in RACE_ORDER[1:]:
            x[f"race_f{lab}"] = np.where(data["race"].isna(), np.nan, (data["race"] == lab).astype(float))
    # College+ is reference.
    for lab in EDUC_ORDER[:-1]:
        x[f"educ_f{lab}"] = np.where(data["education"].isna(), np.nan, (data["education"] == lab).astype(float))
    x["INDFMPIR"] = data["INDFMPIR"]
    x["ever_smoker"] = data["ever_smoker"]
    return x


def survey_logit_matrix(
    data: pd.DataFrame,
    outcome: str,
    predictors: pd.DataFrame,
    domain: pd.Series,
    target: Optional[str] = None,
    max_iter: int = 100,
) -> Dict[str, object]:
    """Survey-weighted logistic regression with Taylor-linearized covariance."""
    des = data.loc[data["PFAS_WT"] > 0].copy()
    Xdf = predictors.reindex(des.index).copy()
    y = des[outcome]
    dom = domain.reindex(des.index).fillna(False)
    I = dom & y.notna() & Xdf.notna().all(axis=1)
    if I.sum() == 0:
        raise ValueError("No complete observations for survey logistic model.")

    ddX = Xdf.loc[I]
    names = ["(Intercept)"] + list(ddX.columns)
    X = np.column_stack([np.ones(len(ddX)), ddX.to_numpy(float)])
    yy = y.loc[I].to_numpy(float)
    w = des.loc[I, "PFAS_WT"].to_numpy(float)

    b = np.zeros(X.shape[1], dtype=float)
    converged = False
    for _ in range(max_iter):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        score = X.T @ (w * (yy - p))
        A = X.T @ ((w * p * (1 - p))[:, None] * X)
        step = np.linalg.pinv(A) @ score
        b += step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    if not converged:
        warnings.warn("Survey logistic model reached maximum iterations without strict convergence.")

    eta = X @ b
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    A = X.T @ ((w * p * (1 - p))[:, None] * X)
    score_i = (w * (yy - p))[:, None] * X

    score_cols = []
    for k in range(X.shape[1]):
        c = f"_score_{k}"
        score_cols.append(c)
        des[c] = 0.0
        des.loc[I, c] = score_i[:, k]

    B = stratified_cluster_variance(des, score_cols)
    Ainv = np.linalg.pinv(A)
    V = Ainv @ B @ Ainv.T
    ses = np.sqrt(np.maximum(np.diag(V), 0.0))
    df = design_df(des)
    crit = t.ppf(0.975, df)

    rows = []
    for name, beta, se in zip(names, b, ses):
        stat = beta / se if se > 0 else np.nan
        pval = 2 * t.sf(abs(stat), df) if np.isfinite(stat) else np.nan
        rows.append({
            "Variable": name,
            "Beta": beta,
            "SE": se,
            "OR": math.exp(beta),
            "Lower95": math.exp(beta - crit * se),
            "Upper95": math.exp(beta + crit * se),
            "p": pval,
        })
    coef = pd.DataFrame(rows)

    out: Dict[str, object] = {"coef": coef, "n": int(I.sum()), "df": df, "domain_n": int(dom.sum())}
    if target is not None:
        rr = coef.loc[coef["Variable"].eq(target)]
        if rr.empty:
            raise KeyError(f"Target coefficient {target!r} not found. Available: {coef['Variable'].tolist()}")
        out.update(rr.iloc[0].to_dict())
    return out


def primary_model(
    data: pd.DataFrame,
    exposure: str,
    domain: pd.Series,
    outcome: str = "obese",
    adjusted: bool = True,
    include_race: bool = True,
    include_sex: bool = True,
) -> Dict[str, object]:
    pred = pd.DataFrame(index=data.index)
    pred["ln_PFAS"] = np.where(data[exposure] > 0, np.log(data[exposure]), np.nan)
    if adjusted:
        pred = pd.concat([pred, covariate_frame(data, include_race=include_race, include_sex=include_sex)], axis=1)
    return survey_logit_matrix(data, outcome, pred, domain, target="ln_PFAS")


def interaction_model_white_black(data: pd.DataFrame, exposure: str, domain: pd.Series) -> Dict[str, object]:
    wb_domain = domain & data["race"].isin(["NH White", "NH Black"])
    pred = pd.DataFrame(index=data.index)
    pred["ln_PFAS"] = np.where(data[exposure] > 0, np.log(data[exposure]), np.nan)
    pred["race_fNH Black"] = np.where(data["race"].isna(), np.nan, (data["race"] == "NH Black").astype(float))
    pred = pd.concat([pred, covariate_frame(data, include_race=False, include_sex=True)], axis=1)
    pred["ln_PFAS_x_NH_Black"] = pred["ln_PFAS"] * pred["race_fNH Black"]
    return survey_logit_matrix(data, "obese", pred, wb_domain, target="ln_PFAS_x_NH_Black")


def weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mid-distribution rank in [0,1]."""
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    ranks = np.empty_like(v, dtype=float)
    total = w.sum()
    c = 0.0
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[j] == v[i]:
            j += 1
        wg = w[i:j].sum()
        mid = (c + 0.5 * wg) / total
        ranks[i:j] = mid
        c += wg
        i = j
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    cov = np.sum(w * (x - mx) * (y - my)) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def survey_weighted_spearman(data: pd.DataFrame, variables: Sequence[str], domain: pd.Series) -> pd.DataFrame:
    names = [LABEL_BY_VAR[v] for v in variables]
    out = pd.DataFrame(np.eye(len(variables)), index=names, columns=names, dtype=float)
    for i, a in enumerate(variables):
        for j, b in enumerate(variables[:i]):
            m = domain & data[a].notna() & data[b].notna() & (data[a] > 0) & (data[b] > 0) & (data["PFAS_WT"] > 0)
            dd = data.loc[m, [a, b, "PFAS_WT"]]
            x = weighted_rank(np.log(dd[a].to_numpy(float)), dd["PFAS_WT"].to_numpy(float))
            y = weighted_rank(np.log(dd[b].to_numpy(float)), dd["PFAS_WT"].to_numpy(float))
            r = weighted_corr(x, y, dd["PFAS_WT"].to_numpy(float))
            out.iloc[i, j] = out.iloc[j, i] = r
    return out


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------

def sample_flow(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        positive = d["PFAS_WT"] > 0
        adult = positive & (d["RIDAGEYR"] >= 20) & (~d["pregnant"])
        analytic = d["analytic_domain"]
        rows.append({
            "Cycle": CYCLE_LABELS[cy],
            "Positive_PFAS_weight": int(positive.sum()),
            "Adults_20plus_nonpregnant": int(adult.sum()),
            "Analytic_valid_BMI": int(analytic.sum()),
            "Complete_primary_model": int(d["primary_complete"].sum()),
        })
    return pd.DataFrame(rows)


def table1_characteristics(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        dom = d["primary_complete"]
        n = int(dom.sum())
        for var, label in [("RIDAGEYR", "Age, years"), ("BMXBMI", "BMI, kg/m^2"), ("INDFMPIR", "PIR")]:
            r = survey_mean(d, var, dom)
            rows.append({"Cycle": CYCLE_LABELS[cy], "Characteristic": label, "Type": "mean", "Estimate": r["estimate"], "SE": r["se"], "N": n})
        r = survey_prop(d, "female", dom)
        rows.append({"Cycle": CYCLE_LABELS[cy], "Characteristic": "Female, %", "Type": "percent", "Estimate": 100*r["estimate"], "SE": 100*r["se"], "N": n})

        for lab in RACE_ORDER:
            tmp = d.copy()
            col = "_indicator"
            tmp[col] = np.where(tmp["race"].isna(), np.nan, (tmp["race"] == lab).astype(float))
            r = survey_prop(tmp, col, dom)
            rows.append({"Cycle": CYCLE_LABELS[cy], "Characteristic": f"Race: {lab}, %", "Type": "percent", "Estimate": 100*r["estimate"], "SE": 100*r["se"], "N": n})
        for lab in EDUC_ORDER:
            tmp = d.copy()
            col = "_indicator"
            tmp[col] = np.where(tmp["education"].isna(), np.nan, (tmp["education"] == lab).astype(float))
            r = survey_prop(tmp, col, dom)
            rows.append({"Cycle": CYCLE_LABELS[cy], "Characteristic": f"Education: {lab}, %", "Type": "percent", "Estimate": 100*r["estimate"], "SE": 100*r["se"], "N": n})
        for var, label in [("ever_smoker", "Ever smoker, %"), ("obese", "Obesity (BMI>=30), %")]:
            r = survey_prop(d, var, dom)
            rows.append({"Cycle": CYCLE_LABELS[cy], "Characteristic": label, "Type": "percent", "Estimate": 100*r["estimate"], "SE": 100*r["se"], "N": n})
    return pd.DataFrame(rows)


def detection_table(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        dom = d["analytic_domain"]
        for var, comment, label in DETECTION_MAP:
            m = dom & d[var].notna()
            if comment in d.columns:
                detected = d.loc[m, comment].eq(0)
            else:
                # NHANES stores below-LOD values as LOD/sqrt(2).
                detected = d.loc[m, var] > (LOD / math.sqrt(2) + 1e-10)
            unweighted = 100 * detected.mean() if len(detected) else np.nan
            tmp = d.copy()
            tmp["_detected"] = np.nan
            tmp.loc[m, "_detected"] = detected.astype(float).to_numpy()
            wr = survey_prop(tmp, "_detected", dom)
            rows.append({
                "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label, "LOD_ng_mL": LOD,
                "Detection_percent_unweighted": unweighted,
                "Detection_percent_weighted": 100*wr["estimate"],
                "N": int(m.sum()),
            })
    return pd.DataFrame(rows)


def geometric_means(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        dom = d["analytic_domain"]
        for var, label in GM_ANALYTES + [("Total_PFOS", "Total PFOS"), ("Total_PFOA", "Total PFOA")]:
            r = survey_gm(d, var, dom)
            rows.append({
                "CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                "GM": r["estimate"], "Lower95": r["lower"], "Upper95": r["upper"],
                "SE_log": r["se_log"], "N": r["n"],
            })
    gm = pd.DataFrame(rows)
    return gm


def geometric_mean_comparison(gm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, label in GM_ANALYTES + [("Total_PFOS", "Total PFOS"), ("Total_PFOA", "Total PFOA")]:
        a = gm[(gm.CycleCode == "J") & (gm.Variable == var)].iloc[0]
        b = gm[(gm.CycleCode == "L") & (gm.Variable == var)].iloc[0]
        rows.append({
            "Variable": var, "PFAS_Compound": label,
            "GM_J": a.GM, "J_Lower95": a.Lower95, "J_Upper95": a.Upper95,
            "GM_L": b.GM, "L_Lower95": b.Lower95, "L_Upper95": b.Upper95,
            "Percent_change_L_vs_J": 100*(b.GM/a.GM - 1),
        })
    return pd.DataFrame(rows)


def run_primary_models(cycles: Mapping[str, pd.DataFrame], adjusted: bool) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        for var, label in MODEL_ANALYTES:
            r = primary_model(d, var, d["analytic_domain"], adjusted=adjusted)
            rows.append({
                "CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                "OR": r["OR"], "Lower95": r["Lower95"], "Upper95": r["Upper95"], "p": r["p"],
                "Beta": r["Beta"], "SE": r["SE"], "N": r["n"], "df": r["df"],
            })
    return pd.DataFrame(rows)


def full_covariate_table(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        pred = pd.DataFrame(index=d.index)
        pred["ln_n_PFOS"] = np.where(d["LBXNFOS"] > 0, np.log(d["LBXNFOS"]), np.nan)
        pred = pd.concat([pred, covariate_frame(d, include_race=True, include_sex=True)], axis=1)
        fit = survey_logit_matrix(d, "obese", pred, d["analytic_domain"])
        c = fit["coef"].copy()
        c.insert(0, "Cycle", CYCLE_LABELS[cy])
        c["N"] = fit["n"]
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def race_stratified_and_interactions(cycles: Mapping[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    int_rows = []
    for cy, d in cycles.items():
        for var, label in MODEL_ANALYTES:
            for race in ["NH White", "NH Black"]:
                dom = d["analytic_domain"] & d["race"].eq(race)
                r = primary_model(d, var, dom, adjusted=True, include_race=False, include_sex=True)
                rows.append({
                    "CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                    "Race": race, "OR": r["OR"], "Lower95": r["Lower95"], "Upper95": r["Upper95"],
                    "p": r["p"], "N": r["n"],
                })
            inter = interaction_model_white_black(d, var, d["analytic_domain"])
            int_rows.append({
                "CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                "Interaction_Beta": inter["Beta"], "SE": inter["SE"], "p_interaction": inter["p"], "N": inter["n"],
            })
    return pd.DataFrame(rows), pd.DataFrame(int_rows)


def sensitivity_models(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    # Supplement Table S3 uses the eight individual compounds (excluding branched PFOA and totals).
    analytes = CORR_ANALYTES
    for cy, d in cycles.items():
        for var, label in analytes:
            # Severe obesity: same covariate set as primary model.
            r = primary_model(d, var, d["analytic_domain"], outcome="severe_obese", adjusted=True)
            rows.append({"Sensitivity_Analysis": "Severe", "PFAS_Compound": label, "Variable": var,
                         "Cycle": CYCLE_LABELS[cy], "OR": r["OR"], "Lower95": r["Lower95"], "Upper95": r["Upper95"], "p": r["p"], "N": r["n"]})

            # Women only: sex is constant, so it is omitted while all other primary covariates remain.
            women_dom = d["analytic_domain"] & d["female"].eq(1)
            r = primary_model(d, var, women_dom, outcome="obese", adjusted=True, include_race=True, include_sex=False)
            rows.append({"Sensitivity_Analysis": "Women", "PFAS_Compound": label, "Variable": var,
                         "Cycle": CYCLE_LABELS[cy], "OR": r["OR"], "Lower95": r["Lower95"], "Upper95": r["Upper95"], "p": r["p"], "N": r["n"]})

            # Exposure-specific 99th percentile exclusion within the analytic domain.
            base = d["analytic_domain"] & d[var].notna() & (d[var] > 0)
            cutoff = float(np.nanpercentile(d.loc[base, var], 99))
            no_outlier_dom = d["analytic_domain"] & d[var].le(cutoff)
            r = primary_model(d, var, no_outlier_dom, outcome="obese", adjusted=True)
            rows.append({"Sensitivity_Analysis": "No Outlier", "PFAS_Compound": label, "Variable": var,
                         "Cycle": CYCLE_LABELS[cy], "OR": r["OR"], "Lower95": r["Lower95"], "Upper95": r["Upper95"], "p": r["p"], "N": r["n"], "Cutoff_99th": cutoff})
    return pd.DataFrame(rows)


def quartile_models(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        for var, label in MODEL_ANALYTES:
            base = d["analytic_domain"] & d[var].notna() & (d[var] > 0)
            q = weighted_quartile_codes(d[var].where(base), d["PFAS_WT"].where(base))
            pred = pd.DataFrame(index=d.index)
            for k in [2,3,4]:
                pred[f"Q{k}"] = np.where(q.isna(), np.nan, (q == k).astype(float))
            pred = pd.concat([pred, covariate_frame(d, include_race=True, include_sex=True)], axis=1)
            fit = survey_logit_matrix(d, "obese", pred, d["analytic_domain"])
            co = fit["coef"].set_index("Variable")
            rows.append({"CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                         "Quartile": "Q1", "OR": 1.0, "Lower95": 1.0, "Upper95": 1.0, "p": np.nan, "N": fit["n"]})
            for k in [2,3,4]:
                rr = co.loc[f"Q{k}"]
                rows.append({"CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Variable": var, "PFAS_Compound": label,
                             "Quartile": f"Q{k}", "OR": rr.OR, "Lower95": rr.Lower95, "Upper95": rr.Upper95, "p": rr.p, "N": fit["n"]})
            # Ordinal trend test.
            trend = pd.DataFrame(index=d.index)
            trend["quartile_ordinal"] = q.astype(float)
            trend = pd.concat([trend, covariate_frame(d, include_race=True, include_sex=True)], axis=1)
            tr = survey_logit_matrix(d, "obese", trend, d["analytic_domain"], target="quartile_ordinal")
            for row in rows[-4:]:
                row["p_trend"] = tr["p"]
    return pd.DataFrame(rows)


def correlation_tables(cycles: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    variables = [v for v, _ in CORR_ANALYTES]
    return {cy: survey_weighted_spearman(d, variables, d["analytic_domain"]) for cy, d in cycles.items()}


def obesity_prevalence_by_race(cycles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cy, d in cycles.items():
        for race in RACE_ORDER:
            dom = d["analytic_domain"] & d["race"].eq(race)
            r = survey_prop(d, "obese", dom)
            rows.append({"CycleCode": cy, "Cycle": CYCLE_LABELS[cy], "Race": race,
                         "Prevalence": r["estimate"], "Lower95": r["lower"], "Upper95": r["upper"], "SE": r["se"], "N": r["n"]})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_figure1_forest(adjusted: pd.DataFrame, out: Path) -> None:
    labels = [lab for _, lab in MODEL_ANALYTES]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    offset = 0.12
    for i, cy in enumerate(["J", "L"]):
        dd = adjusted[adjusted.CycleCode.eq(cy)].set_index("PFAS_Compound").loc[labels]
        yy = y + (-offset if i == 0 else offset)
        x = dd.OR.to_numpy(float)
        lo = x - dd.Lower95.to_numpy(float)
        hi = dd.Upper95.to_numpy(float) - x
        ax.errorbar(x, yy, xerr=[lo, hi], fmt="o" if i == 0 else "s", capsize=3, label=CYCLE_LABELS[cy])
        for xi, yi, p in zip(x, yy, dd.p):
            if p < 0.05:
                ax.text(xi, yi - 0.16, "*", ha="center", va="center", fontsize=12)
    ax.axvline(1.0, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted odds ratio per 1-unit increase in ln(PFAS)")
    ax.set_title("Adjusted PFAS–obesity associations by NHANES cycle")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    savefig(fig, out / "Figure1_Forest_Plot.png")


def main_figure2_concentration(gm_comp: pd.DataFrame, out: Path) -> None:
    # Use the nine individual harmonized laboratory analytes in the manuscript table.
    labels = [lab for _, lab in GM_ANALYTES]
    dd = gm_comp.set_index("PFAS_Compound").loc[labels]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    off = 0.12
    for i, (prefix, lab) in enumerate([("J", "2017-2018"), ("L", "2021-2023")]):
        x = dd[f"GM_{prefix}"].to_numpy(float)
        lo = x - dd[f"{prefix}_Lower95"].to_numpy(float)
        hi = dd[f"{prefix}_Upper95"].to_numpy(float) - x
        yy = y + (-off if i == 0 else off)
        ax.errorbar(x, yy, xerr=[lo, hi], fmt="o" if i == 0 else "s", capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Survey-weighted geometric mean serum concentration (ng/mL; log scale)")
    ax.set_title("Serum PFAS concentration trends across NHANES cycles")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    # Annotate percent change at the right edge in axes coordinates.
    for yi, pct in zip(y, dd["Percent_change_L_vs_J"]):
        ax.text(0.99, yi, f"{pct:+.1f}%", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    savefig(fig, out / "Figure2_Concentration_Trends.png")


def main_figure3_race(race_models: pd.DataFrame, interactions: pd.DataFrame, out: Path) -> None:
    dd = race_models[(race_models.CycleCode == "L") & race_models.Variable.isin(RACE_FIG_ANALYTES)].copy()
    labels = [LABEL_BY_VAR[v] for v in RACE_FIG_ANALYTES]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    offset = 0.12
    for i, race in enumerate(["NH White", "NH Black"]):
        r = dd[dd.Race.eq(race)].set_index("PFAS_Compound").loc[labels]
        yy = y + (-offset if i == 0 else offset)
        x = r.OR.to_numpy(float)
        ax.errorbar(x, yy, xerr=[x-r.Lower95.to_numpy(float), r.Upper95.to_numpy(float)-x],
                    fmt="o" if i == 0 else "s", capsize=3, label=race)
    ax.axvline(1, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted odds ratio per 1-unit increase in ln(PFAS)")
    ax.set_title("Race-stratified PFAS–obesity associations, NHANES 2021–2023")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    ints = interactions[(interactions.CycleCode == "L") & interactions.Variable.isin(["LBXNFOA", "Total_PFOA"])].set_index("PFAS_Compound")
    text = []
    for lab in ["n-PFOA", "Total PFOA"]:
        if lab in ints.index:
            text.append(f"Race × {lab}: p={ints.loc[lab, 'p_interaction']:.3f}")
    ax.text(0.98, 0.02, "\n".join(text), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    savefig(fig, out / "Figure3_Race_Stratified.png")


def main_figure4_conceptual(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    boxes = [
        (0.5, 4.8, 2.2, 1.0, "Upstream exposure determinants\n(water, occupation, products)"),
        (0.5, 2.8, 2.2, 1.0, "Social/structural context\n(neighborhood, resources, stress)"),
        (3.7, 3.7, 2.1, 1.15, "Serum PFAS\nconcentration"),
        (7.0, 4.8, 2.2, 1.0, "Metabolic pathways\n(PPAR, lipid/glucose signaling)"),
        (9.6, 3.7, 1.9, 1.15, "Obesity\n(BMI ≥30)"),
        (6.9, 2.1, 2.4, 1.0, "Measured covariates\nage, sex, education, PIR, smoking"),
    ]
    for x,y,w,h,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=.04,rounding_size=.08", fill=False, lw=1.4))
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=9)
    arrows = [((2.7,5.3),(3.7,4.35)), ((2.7,3.3),(3.7,4.1)), ((5.8,4.3),(7.0,5.25)), ((9.2,5.25),(9.6,4.35)), ((5.8,4.0),(9.6,4.05)), ((8.1,3.1),(10.1,3.7))]
    for a,b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle":"->", "lw":1.2})
    ax.text(5.8, 6.35, "Race/ethnicity as a social classification and potential effect modifier", ha="center", fontsize=10)
    ax.annotate("", xy=(5.0,4.85), xytext=(5.8,6.1), arrowprops={"arrowstyle":"->", "lw":1.0, "ls":"--"})
    ax.annotate("", xy=(10.3,4.9), xytext=(6.4,6.1), arrowprops={"arrowstyle":"->", "lw":1.0, "ls":"--"})
    ax.text(5.9, 0.65, "Cross-sectional NHANES analysis: associations do not establish temporal or causal direction", ha="center", style="italic", fontsize=9)
    savefig(fig, out / "Figure4_Conceptual_Framework.png")


def supp_figure_s1_distributions(cycles: Mapping[str, pd.DataFrame], out: Path) -> None:
    """
    Supplementary Figure S1: race/ethnicity-specific PFAS distributions in both
    NHANES cycles. The layout is intentionally 4 x 3 rather than 2 x 6 so the
    category labels remain readable at journal-column width.

    Natural-log transformation is used here for consistency with the manuscript's
    regression scale. If a log2 descriptive display is preferred, replace np.log
    with np.log2 and update the figure caption accordingly.
    """
    wrapped_race = [
        "NH White",
        "NH Black",
        "Mexican\nAmerican",
        "Other\nHispanic",
        "NH Asian",
        "Other/\nMulti",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True)

    for cycle_index, cy in enumerate(["J", "L"]):
        d = cycles[cy]
        base_row = cycle_index * 2

        for k, var in enumerate(DISTRIBUTION_ANALYTES):
            row = base_row + (k // 3)
            col = k % 3
            ax = axes[row, col]

            vals = []
            for race in RACE_ORDER:
                m = d["analytic_domain"] & d["race"].eq(race) & d[var].gt(0)
                vals.append(np.log(d.loc[m, var].to_numpy(float)))

            ax.boxplot(
                vals,
                widths=0.55,
                patch_artist=False,
                showfliers=True,
                medianprops={"linewidth": 1.8},
                whiskerprops={"linewidth": 1.1},
                capprops={"linewidth": 1.1},
                boxprops={"linewidth": 1.2},
                flierprops={"markersize": 2.6, "alpha": 0.65},
            )

            ax.set_title(LABEL_BY_VAR[var], fontsize=11, pad=7)
            ax.set_xticks(np.arange(1, len(RACE_ORDER) + 1))
            ax.set_xticklabels(wrapped_race, rotation=24, ha="right", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.18)

            if col == 0:
                ax.set_ylabel("ln serum PFAS (ng/mL)", fontsize=9)

        # Cycle headings placed above each 2-row panel.
        panel_label = "A" if cy == "J" else "B"
        axes[base_row, 0].text(
            -0.16, 1.26,
            f"{panel_label}. NHANES {CYCLE_LABELS[cy]}",
            transform=axes[base_row, 0].transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold",
        )

    fig.suptitle(
        "Supplementary Figure S1. Serum PFAS distributions by race/ethnicity",
        fontsize=14, y=1.015,
    )

    savefig(fig, out / "Figure_S1_PFAS_Distributions_by_Race.png")


def supp_figure_s2_obesity(prev: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    x = np.arange(len(RACE_ORDER)); width = 0.34
    for i, cy in enumerate(["J", "L"]):
        dd = prev[prev.CycleCode.eq(cy)].set_index("Race").loc[RACE_ORDER]
        pos = x + (-width/2 if i == 0 else width/2)
        y = 100*dd.Prevalence.to_numpy(float)
        lo = 100*(dd.Prevalence-dd.Lower95).to_numpy(float)
        hi = 100*(dd.Upper95-dd.Prevalence).to_numpy(float)
        ax.bar(pos, y, width=width, label=CYCLE_LABELS[cy])
        ax.errorbar(pos, y, yerr=[lo, hi], fmt="none", capsize=3)
        for xx, yy in zip(pos, y):
            ax.text(xx, yy+1.5, f"{yy:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(RACE_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Survey-weighted obesity prevalence (%)")
    ax.set_title("Supplementary Figure S2. Obesity prevalence by race/ethnicity and NHANES cycle")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    savefig(fig, out / "Figure_S2_Obesity_Prevalence_by_Race.png")


def supp_figure_s3_dose_response(quartiles: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, var in zip(axes, DOSE_RESPONSE_ANALYTES):
        lab = LABEL_BY_VAR[var]
        for i, cy in enumerate(["J", "L"]):
            dd = quartiles[(quartiles.CycleCode == cy) & (quartiles.Variable == var)].copy()
            dd["q"] = dd.Quartile.str.replace("Q", "", regex=False).astype(int)
            dd = dd.sort_values("q")
            ax.plot(dd.q, dd.OR, marker="o", ls="-" if cy == "J" else "--", label=CYCLE_LABELS[cy])
        ax.axhline(1, ls=":", lw=1)
        ax.set_title(lab)
        ax.set_xticks([1,2,3,4]); ax.set_xticklabels(["Q1","Q2","Q3","Q4"])
        ax.set_ylabel("Adjusted OR")
        jtrend = quartiles[(quartiles.CycleCode == "J") & (quartiles.Variable == var)].p_trend.dropna()
        if len(jtrend):
            ax.text(0.03, 0.04, f"J trend p={jtrend.iloc[0]:.3f}", transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=0.15)
    axes[-1].axis("off")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Supplementary Figure S3. Dose-response: adjusted OR by PFAS quartile (Q1 reference)")
    fig.subplots_adjust(bottom=0.12)
    savefig(fig, out / "Figure_S3_Dose_Response_Curves.png")


def supp_figure_s4_correlations(corr: Mapping[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    vmin, vmax = -1, 1
    im = None
    for ax, cy in zip(axes, ["J", "L"]):
        mat = corr[cy]
        im = ax.imshow(mat.to_numpy(float), vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title(CYCLE_LABELS[cy])
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                ax.text(j, i, f"{mat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Weighted Spearman ρ")
    fig.suptitle("Supplementary Figure S4. PFAS Spearman correlation heatmaps (survey-weighted)")
    fig.subplots_adjust(right=0.91, top=0.86, bottom=0.2)
    fig.savefig(out / "Figure_S4_PFAS_Correlation_Heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def lowess_line(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 8:
        return np.array([]), np.array([])
    order = np.argsort(x)
    x, y = x[order], y[order]
    if lowess is None:
        b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        return xx, np.polyval(b, xx)
    sm = lowess(y, x, frac=0.45, return_sorted=True)
    return sm[:,0], sm[:,1]


def supp_figure_s5_scatter(cycles: Mapping[str, pd.DataFrame], out: Path) -> None:
    d = cycles["L"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, var in zip(axes.ravel(), SCATTER_ANALYTES):
        lab = LABEL_BY_VAR[var]
        for race in RACE_ORDER:
            m = d["analytic_domain"] & d["race"].eq(race) & d[var].gt(0) & d["BMXBMI"].notna()
            x = np.log2(d.loc[m, var].to_numpy(float)); y = d.loc[m, "BMXBMI"].to_numpy(float)
            ax.scatter(x, y, s=8, alpha=0.35, label=race)
            xx, yy = lowess_line(x, y)
            if len(xx): ax.plot(xx, yy, lw=1)
        ax.set_title(lab); ax.set_xlabel("log2(ng/mL)"); ax.set_ylabel("BMI (kg/m²)"); ax.grid(alpha=0.15)
    handles, labs = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Supplementary Figure S5. PFAS vs BMI by race/ethnicity, NHANES 2021–2023")
    fig.subplots_adjust(bottom=0.12)
    savefig(fig, out / "Figure_S5_PFAS_BMI_Scatter_by_Race.png")


def supp_figure_s6_dag(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    nodes = {
        "PFAS Exposure": (1.3, 4.3), "Obesity\n(BMI ≥30)": (10.2, 4.3), "NHANES\nCycle": (5.8, 4.3),
        "Age": (3.2, 6.6), "Sex": (5.4, 6.6), "Race/\nEthnicity": (8.0, 6.6),
        "Education": (3.1, 1.6), "Income\n(PIR)": (5.8, 1.6), "Smoking": (8.2, 1.6),
    }
    for name,(x,y) in nodes.items():
        circ = plt.Circle((x,y), 0.65 if "PFAS" in name or "Obesity" in name else 0.5, fill=False, lw=1.5)
        ax.add_patch(circ); ax.text(x,y,name,ha="center",va="center",fontsize=9)
    def arr(a,b,ls="-",lw=1.2):
        ax.annotate("", xy=nodes[b], xytext=nodes[a], arrowprops={"arrowstyle":"->","lw":lw,"ls":ls,"shrinkA":28,"shrinkB":28})
    for a,b in [("Age","PFAS Exposure"),("Age","Obesity\n(BMI ≥30)"),("Sex","PFAS Exposure"),("Sex","Obesity\n(BMI ≥30)"),
                ("Education","PFAS Exposure"),("Education","Obesity\n(BMI ≥30)"),("Income\n(PIR)","PFAS Exposure"),("Income\n(PIR)","Obesity\n(BMI ≥30)"),
                ("Smoking","PFAS Exposure"),("Smoking","Obesity\n(BMI ≥30)"),("NHANES\nCycle","PFAS Exposure"),("NHANES\nCycle","Obesity\n(BMI ≥30)")]:
        arr(a,b)
    arr("Race/\nEthnicity","PFAS Exposure",ls="--"); arr("Race/\nEthnicity","Obesity\n(BMI ≥30)",ls="--")
    ax.annotate("", xy=nodes["Obesity\n(BMI ≥30)"], xytext=nodes["PFAS Exposure"], arrowprops={"arrowstyle":"->","lw":2.3,"shrinkA":42,"shrinkB":42})
    ax.set_title("Supplementary Figure S6. Directed acyclic graph (DAG): PFAS–obesity conceptual framework")
    savefig(fig, out / "Figure_S6_DAG.png")


# -----------------------------------------------------------------------------
# Validation and output helpers
# -----------------------------------------------------------------------------

def validation_report(cycles: Mapping[str, pd.DataFrame], adjusted: pd.DataFrame, gm: pd.DataFrame,
                      interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cy in ["J", "L"]:
        rows.append({"Section":"Sample N", "Cycle":cy, "Item":"Analytic N", "Expected":EXPECTED_N[cy]["analytic"],
                     "Observed":int(cycles[cy]["analytic_domain"].sum())})
        rows.append({"Section":"Sample N", "Cycle":cy, "Item":"Complete N", "Expected":EXPECTED_N[cy]["complete"],
                     "Observed":int(cycles[cy]["primary_complete"].sum())})
        for lab, vals in EXPECTED_ADJUSTED_OR[cy].items():
            obs = adjusted[(adjusted.CycleCode==cy)&(adjusted.PFAS_Compound==lab)].iloc[0]
            for field, exp in zip(["OR","Lower95","Upper95","p"], vals):
                rows.append({"Section":"Adjusted model", "Cycle":cy, "Item":f"{lab} {field}", "Expected":exp, "Observed":float(obs[field])})
        for lab, vals in EXPECTED_GM[cy].items():
            obs = gm[(gm.CycleCode==cy)&(gm.PFAS_Compound==lab)].iloc[0]
            for field, exp in zip(["GM","Lower95","Upper95"], vals):
                rows.append({"Section":"Geometric mean", "Cycle":cy, "Item":f"{lab} {field}", "Expected":exp, "Observed":float(obs[field])})
    for lab, exp in EXPECTED_INTERACTION_P.items():
        obs = interactions[(interactions.CycleCode=="L")&(interactions.PFAS_Compound==lab)].iloc[0]
        rows.append({"Section":"Interaction", "Cycle":"L", "Item":f"Race x {lab} p", "Expected":exp, "Observed":float(obs.p_interaction)})
    out = pd.DataFrame(rows)
    out["Absolute_Difference"] = (pd.to_numeric(out.Expected) - pd.to_numeric(out.Observed)).abs()
    out["Relative_Difference"] = out.Absolute_Difference / pd.to_numeric(out.Expected).abs().replace(0, np.nan)
    out["Within_rounding_tolerance"] = out.Absolute_Difference <= np.where(out.Section.eq("Sample N"), 0, 0.006)
    return out


def copy_aliases(fig_dir: Path) -> None:
    """Create a few aliases matching common manuscript/supplement naming variants."""
    aliases = {
        "Figure_S1_PFAS_Distributions_by_Race.png": ["Supplementary_Figure_S1.png", "FigureS1_PFAS_Distributions_by_Race.png"],
        "Figure_S2_Obesity_Prevalence_by_Race.png": ["Supplementary_Figure_S2.png", "FigureS2_Obesity_Prevalence_by_Race.png"],
        "Figure_S3_Dose_Response_Curves.png": ["Supplementary_Figure_S3.png", "FigureS3_Dose_Response_Curves.png"],
        "Figure_S4_PFAS_Correlation_Heatmaps.png": ["Supplementary_Figure_S4.png", "FigureS4_PFAS_Correlation_Heatmaps.png"],
        "Figure_S5_PFAS_BMI_Scatter_by_Race.png": ["Supplementary_Figure_S5.png", "FigureS5_PFAS_BMI_Scatter_by_Race.png"],
        "Figure_S6_DAG.png": ["Supplementary_Figure_S6.png", "FigureS6_DAG.png"],
    }
    for src, names in aliases.items():
        for name in names:
            shutil.copyfile(fig_dir/src, fig_dir/name)


def save_tables(
    table_dir: Path,
    flow: pd.DataFrame,
    char: pd.DataFrame,
    detection: pd.DataFrame,
    gm: pd.DataFrame,
    gm_comp: pd.DataFrame,
    crude: pd.DataFrame,
    adjusted: pd.DataFrame,
    full_covars: pd.DataFrame,
    race_models: pd.DataFrame,
    interactions: pd.DataFrame,
    sensitivity: pd.DataFrame,
    quartiles: pd.DataFrame,
    corr: Mapping[str, pd.DataFrame],
    obesity_race: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    table_dir = ensure_dir(table_dir)
    flow.to_csv(table_dir/"Sample_Flow.csv", index=False)
    char.to_csv(table_dir/"Table1_Weighted_Sample_Characteristics.csv", index=False)
    detection.to_csv(table_dir/"Table_S4_PFAS_Detection_Frequencies.csv", index=False)
    gm.to_csv(table_dir/"PFAS_Geometric_Means_All.csv", index=False)
    gm_comp.to_csv(table_dir/"Table3_PFAS_Geometric_Means_by_Cycle.csv", index=False)
    crude[crude["Variable"].isin([v for v, _ in CORR_ANALYTES])].to_csv(
        table_dir/"Table_S1_Crude_PFAS_Obesity_ORs.csv", index=False
    )
    crude.to_csv(table_dir/"Crude_PFAS_Obesity_ORs_All_Analytes.csv", index=False)
    adjusted.to_csv(table_dir/"Table2_Adjusted_PFAS_Obesity_ORs.csv", index=False)
    full_covars.to_csv(table_dir/"Table_S5_Full_Model_Covariate_ORs.csv", index=False)
    race_models.to_csv(table_dir/"Race_Stratified_Models.csv", index=False)
    interactions.to_csv(table_dir/"Race_PFAS_Interaction_Tests.csv", index=False)
    sensitivity.to_csv(table_dir/"Table_S3_Sensitivity_Analyses.csv", index=False)
    quartiles.to_csv(table_dir/"Quartile_Dose_Response_Models.csv", index=False)
    corr["J"].to_csv(table_dir/"Table_S2a_PFAS_Spearman_Correlations_J.csv")
    corr["L"].to_csv(table_dir/"Table_S2b_PFAS_Spearman_Correlations_L.csv")
    obesity_race.to_csv(table_dir/"Obesity_Prevalence_by_Race.csv", index=False)
    validation.to_csv(table_dir/"Validation_Against_Manuscript.csv", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(data_dir: Path, output_dir: Path, allow_download: bool = True) -> None:
    data_dir = ensure_dir(data_dir)
    output_dir = ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    table_dir = ensure_dir(output_dir / "tables")

    files = ensure_data_files(data_dir, allow_download=allow_download)
    cycles = {
        cy: prep_cycle(merge_cycle(files[cy]), cy)
        for cy in ["J", "L"]
    }

    flow = sample_flow(cycles)
    char = table1_characteristics(cycles)
    detection = detection_table(cycles)
    gm = geometric_means(cycles)
    gm_comp = geometric_mean_comparison(gm)
    crude = run_primary_models(cycles, adjusted=False)
    adjusted = run_primary_models(cycles, adjusted=True)
    full_covars = full_covariate_table(cycles)
    race_models, interactions = race_stratified_and_interactions(cycles)
    sensitivity = sensitivity_models(cycles)
    quartiles = quartile_models(cycles)
    corr = correlation_tables(cycles)
    obesity_race = obesity_prevalence_by_race(cycles)
    validation = validation_report(cycles, adjusted, gm, interactions)

    main_figure1_forest(adjusted, fig_dir)
    main_figure2_concentration(gm_comp, fig_dir)
    main_figure3_race(race_models, interactions, fig_dir)
    main_figure4_conceptual(fig_dir)
    supp_figure_s1_distributions(cycles, fig_dir)
    supp_figure_s2_obesity(obesity_race, fig_dir)
    supp_figure_s3_dose_response(quartiles, fig_dir)
    supp_figure_s4_correlations(corr, fig_dir)
    supp_figure_s5_scatter(cycles, fig_dir)
    supp_figure_s6_dag(fig_dir)
    copy_aliases(fig_dir)

    save_tables(table_dir, flow, char, detection, gm, gm_comp, crude, adjusted,
                full_covars, race_models, interactions, sensitivity, quartiles,
                corr, obesity_race, validation)

    summary = {
        "J_analytic_n": int(cycles["J"]["analytic_domain"].sum()),
        "J_complete_n": int(cycles["J"]["primary_complete"].sum()),
        "L_analytic_n": int(cycles["L"]["analytic_domain"].sum()),
        "L_complete_n": int(cycles["L"]["primary_complete"].sum()),
        "validation_rows": int(len(validation)),
        "validation_rows_within_tolerance": int(validation.Within_rounding_tolerance.sum()),
        "J_weight": WEIGHT_COL["J"],
        "L_weight": WEIGHT_COL["L"],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nAnalysis complete.")
    print(json.dumps(summary, indent=2))
    print(f"Figures: {fig_dir}")
    print(f"Tables:  {table_dir}")
    print("Review tables/Validation_Against_Manuscript.csv before submission.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the JFTH NHANES PFAS-obesity analysis and figures.")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing NHANES XPT files.")
    p.add_argument("--output-dir", type=Path, default=Path("JFTH_PFAS_results"), help="Output directory.")
    p.add_argument("--no-download", action="store_true", help="Do not attempt CDC download if a data file is missing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, allow_download=not args.no_download)
