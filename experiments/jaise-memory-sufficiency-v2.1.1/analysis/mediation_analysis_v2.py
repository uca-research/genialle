"""Mediation analysis of the C4 operational advantage by token composition.

Reproduces the mediation paragraph of the journal article (Section 6.2):
re-fits the block-adjusted fixed-effects model for dynamic GPU energy and
wall-clock time, adding completion tokens (and prompt tokens) as covariates,
to test whether the C4-C0 operational differences are accounted for by
completion length ("generation regularisation") rather than by a residual
condition effect.

Data: reads only the published final-run episode log
(data/final_run/episodes.jsonl.gz). No experimental re-execution is needed.

Analysis-only dependencies (not part of the experimental runtime frozen in
environment/requirements_frozen.txt): pandas, statsmodels, scipy.

Expected key outputs (full run, 450 episodes, 90 blocks):
  - Sanity check reproducing Table 7 means and the Table 8 paired
    C4-C0 differences (-1.356 s, -330.1 J).
  - Dynamic energy: C4 coefficient -330.1 J (p < 1e-6) attenuated to
    -7.3 J (p = 0.57) after adding completion tokens.
  - Time: -1.356 s attenuated to +0.035 s (p = 0.45).
  - With prompt and completion tokens included, the joint condition
    F-test is not detectable (p = 0.11 energy, p = 0.53 time).
  - Paired block-level deltas: corr(d_completion, d_energy) = 0.983,
    corr(d_completion, d_time) = 0.988.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data" / "final_run" / "episodes.jsonl.gz"


def load_episodes() -> pd.DataFrame:
    rows = []
    with gzip.open(DATA, "rt") as f:
        for line in f:
            e = json.loads(line)
            rows.append(
                {
                    "condition": e["condition"],
                    "task": e["task_id"],
                    "rep": e["repetition"],
                    "time": e["wall_time_sec"],
                    "dyn_energy": e["power"]["dynamic_energy_j"],
                    "prompt_tok": e["usage"]["prompt_tokens"],
                    "comp_tok": e["usage"]["completion_tokens"],
                }
            )
    df = pd.DataFrame(rows)
    df["block"] = df["task"] + "_r" + df["rep"].astype(str)
    assert len(df) == 450 and df["block"].nunique() == 90
    return df


def condition_ftest(model) -> tuple[float, float]:
    terms = [t for t in model.params.index if t.startswith("C(condition)")]
    R = np.zeros((len(terms), len(model.params)))
    for i, t in enumerate(terms):
        R[i, list(model.params.index).index(t)] = 1
    ft = model.f_test(R)
    return float(ft.fvalue), float(ft.pvalue)


def fit_outcome(df: pd.DataFrame, outcome: str) -> None:
    print(f"\n================ OUTCOME: {outcome} ================")
    specs = [
        ("no tokens (paper model)", f"{outcome} ~ C(condition) + C(block)"),
        ("+ completion tokens", f"{outcome} ~ C(condition) + C(block) + comp_tok"),
        ("+ completion + prompt", f"{outcome} ~ C(condition) + C(block) + comp_tok + prompt_tok"),
    ]
    for name, formula in specs:
        m = smf.ols(formula, data=df).fit()
        c4 = m.params["C(condition)[T.C4]"]
        p4 = m.pvalues["C(condition)[T.C4]"]
        fv, fp = condition_ftest(m)
        print(
            f"  [{name:24s}] C4 vs C0 coef = {c4:9.3f}  p={p4:.2e} | "
            f"condition F={fv:.2f}, p={fp:.2e} | R2={m.rsquared:.3f}"
        )


def paired_decomposition(df: pd.DataFrame) -> None:
    print("\n=== Paired C4-C0 block-level decomposition ===")
    piv = {
        v: df.pivot_table(index="block", columns="condition", values=v)
        for v in ("time", "dyn_energy", "comp_tok")
    }
    d_comp = piv["comp_tok"]["C4"] - piv["comp_tok"]["C0"]
    for label, key, unit in (("energy", "dyn_energy", "J"), ("time", "time", "s")):
        d_out = piv[key]["C4"] - piv[key]["C0"]
        r, p = stats.pearsonr(d_comp, d_out)
        X = sm.add_constant(d_comp.values)
        mm = sm.OLS(d_out.values, X).fit()
        print(
            f"{label}: raw effect = {d_out.mean():.3f} {unit} | "
            f"corr with d(completion tokens) r={r:.3f} (p={p:.2e}) | "
            f"intercept after adjustment = {mm.params[0]:.3f} {unit} "
            f"(p={mm.pvalues[0]:.3f})"
        )


def main() -> None:
    df = load_episodes()
    print("=== Sanity check vs Table 7 (means by condition) ===")
    print(
        df.groupby("condition")[["time", "dyn_energy", "prompt_tok", "comp_tok"]]
        .mean()
        .round(2)
    )
    fit_outcome(df, "dyn_energy")
    fit_outcome(df, "time")
    paired_decomposition(df)


if __name__ == "__main__":
    main()
