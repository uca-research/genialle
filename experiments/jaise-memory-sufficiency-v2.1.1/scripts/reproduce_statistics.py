#!/usr/bin/env python3

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def find_episode_file():
    candidates = [
        ROOT / "data" / "final_run" / "episodes.jsonl.gz",
        ROOT / "data" / "final_run" / "episodes.jsonl",
        ROOT / "data" / "episodes.jsonl.gz",
        ROOT / "data" / "episodes.jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    found = list(ROOT.rglob("episodes.jsonl.gz"))
    if not found:
        found = list(ROOT.rglob("episodes.jsonl"))

    if len(found) == 1:
        return found[0]

    if not found:
        raise FileNotFoundError(
            "Could not locate episodes.jsonl or episodes.jsonl.gz"
        )

    raise RuntimeError(
        "Multiple episode files found. Please specify which one should be used:\n"
        + "\n".join(str(p) for p in found)
    )


def load_episodes(path):
    opener = gzip.open if path.suffix == ".gz" else open

    rows = []

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)

            rows.append({
                "condition": e["condition"],
                "task_id": e["task_id"],
                "repetition": e["repetition"],
                "time_sec": e["wall_time_sec"],
                "dynamic_energy_j": e["power"]["dynamic_energy_j"],
                "prompt_tokens": e["usage"]["prompt_tokens"],
                "completion_tokens": e["usage"]["completion_tokens"],
                "total_tokens": e["usage"]["total_tokens"],
            })

    df = pd.DataFrame(rows)

    df["condition"] = pd.Categorical(
        df["condition"],
        categories=["C0", "C1", "C2", "C3", "C4"],
        ordered=True,
    )

    df["block_id"] = (
        df["task_id"].astype(str)
        + "__r"
        + df["repetition"].astype(str)
    )

    return df


def dummy(series, prefix):
    return pd.get_dummies(
        series,
        prefix=prefix,
        drop_first=True,
        dtype=float,
    )


def fit_ols(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ beta
    sse = float(residuals @ residuals)

    n = len(y)
    rank = int(np.linalg.matrix_rank(X))
    df_resid = n - rank

    sigma2 = sse / df_resid
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))

    tvals = beta / se
    pvals = 2 * t_dist.sf(np.abs(tvals), df_resid)

    return {
        "beta": beta,
        "se": se,
        "t": tvals,
        "p": pvals,
        "sse": sse,
        "rank": rank,
        "df_resid": df_resid,
    }


def nested_f(reduced, full, n):
    df_num = full["rank"] - reduced["rank"]
    df_den = n - full["rank"]

    F = (
        ((reduced["sse"] - full["sse"]) / df_num)
        /
        (full["sse"] / df_den)
    )

    p = f_dist.sf(F, df_num, df_den)

    return F, p, df_num, df_den


episodes_path = find_episode_file()
df = load_episodes(episodes_path)

print("Data:", episodes_path.relative_to(ROOT))
print("Episodes:", len(df))
print("Matched blocks:", df["block_id"].nunique())
print("Trajectories:", df["task_id"].nunique())
print("Conditions:", list(df["condition"].cat.categories))

block_sizes = df.groupby("block_id", observed=True).size()

if len(df) != 450:
    raise RuntimeError(f"Expected 450 episodes, found {len(df)}")

if df["block_id"].nunique() != 90:
    raise RuntimeError(
        f"Expected 90 matched blocks, found {df['block_id'].nunique()}"
    )

if not (block_sizes == 5).all():
    raise RuntimeError("Every matched block must contain five conditions")


# ---------------------------------------------------------------------
# Common design matrices
# ---------------------------------------------------------------------

intercept = pd.DataFrame(
    {"Intercept": np.ones(len(df))},
    index=df.index,
)

condition = dummy(df["condition"], "condition")
task = dummy(df["task_id"], "task")
block = dummy(df["block_id"], "block")

interactions = {}

for c in condition.columns:
    for t in task.columns:
        interactions[f"{c}:{t}"] = condition[c] * task[t]

interaction_df = pd.DataFrame(
    interactions,
    index=df.index,
)


# =====================================================================
# RQ4: matched-block fixed-effects interaction analysis
# =====================================================================

X_rq4_reduced = pd.concat(
    [intercept, block, condition],
    axis=1,
)

X_rq4_full = pd.concat(
    [intercept, block, condition, interaction_df],
    axis=1,
)

rq4_rows = []

rq4_outcomes = [
    "time_sec",
    "dynamic_energy_j",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

for outcome in rq4_outcomes:

    y = df[outcome].to_numpy(float)

    reduced = fit_ols(y, X_rq4_reduced)
    full = fit_ols(y, X_rq4_full)

    F, p, df_num, df_den = nested_f(
        reduced,
        full,
        len(df),
    )

    rq4_rows.append({
        "outcome": outcome,
        "F": F,
        "df_num": df_num,
        "df_den": df_den,
        "p": p,
    })


rq4 = pd.DataFrame(rq4_rows)

rq4.to_csv(
    RESULTS / "rq4_block_fixed_effects.csv",
    index=False,
)

print("\n=== RQ4: CONFIGURATION x TRAJECTORY ===")
print(rq4.to_string(index=False))


# =====================================================================
# Section 5.6: token-composition mechanism analysis
# =====================================================================

def mechanism_model(outcome, covariates, label):

    y = df[outcome].to_numpy(float)

    covariate_frames = [
        df[[name]].astype(float)
        for name in covariates
    ]

    X_reduced = pd.concat(
        [intercept, block] + covariate_frames,
        axis=1,
    )

    X_full = pd.concat(
        [intercept, block, condition] + covariate_frames,
        axis=1,
    )

    reduced = fit_ols(y, X_reduced)
    full = fit_ols(y, X_full)

    F, overall_p, df_num, df_den = nested_f(
        reduced,
        full,
        len(df),
    )

    c4_index = list(X_full.columns).index("condition_C4")

    return {
        "outcome": outcome,
        "model": label,
        "c4_minus_c0": full["beta"][c4_index],
        "c4_p": full["p"][c4_index],
        "overall_condition_F": F,
        "overall_df_num": df_num,
        "overall_df_den": df_den,
        "overall_condition_p": overall_p,
    }


mechanism_rows = []

for outcome in ["dynamic_energy_j", "time_sec"]:

    mechanism_rows.append(
        mechanism_model(
            outcome,
            [],
            "block + condition",
        )
    )

    mechanism_rows.append(
        mechanism_model(
            outcome,
            ["completion_tokens"],
            "block + condition + completion_tokens",
        )
    )

    mechanism_rows.append(
        mechanism_model(
            outcome,
            ["prompt_tokens", "completion_tokens"],
            "block + condition + prompt_tokens + completion_tokens",
        )
    )


mechanism = pd.DataFrame(mechanism_rows)

mechanism.to_csv(
    RESULTS / "mechanism_block_fixed_effects.csv",
    index=False,
)

print("\n=== TOKEN-COMPOSITION MODELS ===")
print(mechanism.to_string(index=False))


# =====================================================================
# Matched-block C4-C0 token-composition correlations
# =====================================================================

wide = df.pivot(
    index="block_id",
    columns="condition",
    values=[
        "completion_tokens",
        "time_sec",
        "dynamic_energy_j",
    ],
)

completion_diff = (
    wide["completion_tokens"]["C4"]
    - wide["completion_tokens"]["C0"]
)

time_diff = (
    wide["time_sec"]["C4"]
    - wide["time_sec"]["C0"]
)

energy_diff = (
    wide["dynamic_energy_j"]["C4"]
    - wide["dynamic_energy_j"]["C0"]
)

corr_rows = []

for name, values in [
    ("dynamic_energy_j", energy_diff),
    ("time_sec", time_diff),
]:
    pearson = pearsonr(completion_diff, values)
    spearman = spearmanr(completion_diff, values)

    corr_rows.append({
        "x": "C4-C0 completion_tokens",
        "y": f"C4-C0 {name}",
        "n": len(values),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    })


correlations = pd.DataFrame(corr_rows)

correlations.to_csv(
    RESULTS / "mechanism_correlations_reproduced.csv",
    index=False,
)

print("\n=== MATCHED-BLOCK CORRELATIONS ===")
print(correlations.to_string(index=False))

print("\nGenerated:")
for name in [
    "rq4_block_fixed_effects.csv",
    "mechanism_block_fixed_effects.csv",
    "mechanism_correlations_reproduced.csv",
]:
    print(" - results/" + name)
