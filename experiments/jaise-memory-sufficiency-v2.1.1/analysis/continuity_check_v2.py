"""Functional manipulation check: was longitudinal memory operationally used?

Reproduces the continuity-check paragraph of the journal article
(Section 5): for each episode, round-3 (round_idx == 2) Teacher and Adapter
final outputs are searched for task-specific lexical markers of
prior-round content that is absent from the round-3 prompt (round-3
learner event and adaptation target, learning goal, learner profile,
and agent system prompts). An episode counts as "memory used" if either
the Teacher or the Adapter round-3 output matches the marker.

Methodological notes (stated in the article):
  - Markers were defined a priori from the trajectory specifications in
    config/tasks_v2.yaml, before inspecting any output text.
  - Two markers proved topically natural and non-discriminative post hoc
    ("training loss"/"misconception" for overfitting_adaptation; the
    two-hours constraint for llm_agents_study_plan, which is rarely
    verbalised under any condition). The "discriminative subset" reported
    in the article (C0 base rate < 15%) is therefore a declared post-hoc
    restriction; the pooled six-task contrast is the primary figure.

Data: reads only the published final-run step log
(data/final_run/steps.jsonl.gz). No experimental re-execution is needed.

Analysis-only dependencies (not part of the experimental runtime frozen in
environment/requirements_frozen.txt): pandas, scipy.

Expected key outputs (full run):
  - Pooled six tasks: marker present in 21.1% of C0 episodes vs 78.9%
    under C4 (74.4-80.0% under C1-C3); McNemar C4 vs C0: 53 vs 1,
    p = 6.1e-15.
  - Discriminative subset (C0 base rate < 15%: gradient_descent,
    cybersecurity, software_architecture, database_normalization):
    5.0% vs 88.3%; McNemar 50 vs 0, p = 1.8e-15.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

DATA = Path(__file__).resolve().parents[1] / "data" / "final_run" / "steps.jsonl.gz"

# Task-specific markers of prior-round (round 1-2) content absent from the
# round-3 prompt. Verified against config/tasks_v2.yaml: none of these
# strings occur in the round-3 learner event, the round-3 adaptation
# target, the learning goal, the learner profile, or the system prompts.
MARKERS: dict[str, dict[str, object]] = {
    "llm_agents_study_plan": {
        "regex": re.compile(r"\btwo[\s-]hours?\b|\b2[\s-]?h(ours?)?\b(?!\d)", re.I),
        "label": "round-2 'two hours per week' constraint (static profile says four)",
    },
    "gradient_descent_scaffolding": {
        "regex": re.compile(
            r"\(x\s*[-\u2212\u2013]\s*3\)|x\s*=\s*3\b|minimum (is |at )?(x\s*=\s*)?3\b",
            re.I,
        ),
        "label": "objective f(x)=(x-3)^2 named in round 1 only",
    },
    "cybersecurity_threat_model": {
        "regex": re.compile(r"\bupload|\bGPU\b", re.I),
        "label": "teacher uploads / shared GPU introduced in round 2",
    },
    "software_architecture_tradeoffs": {
        "regex": re.compile(r"\boutage|\bburst", re.I),
        "label": "outages / bursts introduced in round 2",
    },
    "database_normalization_misconception": {
        "regex": re.compile(r"enrol", re.I),
        "label": "student-course-enrolment schema named in round 1",
    },
    "overfitting_adaptation": {
        "regex": re.compile(r"training loss|misconception", re.I),
        "label": "round-2 'training loss' misconception (topically natural)",
    },
}

DISCRIMINATIVE = [
    "gradient_descent_scaffolding",
    "cybersecurity_threat_model",
    "software_architecture_tradeoffs",
    "database_normalization_misconception",
]

CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]


def load_round3() -> pd.DataFrame:
    rows = []
    with gzip.open(DATA, "rt") as f:
        for line in f:
            s = json.loads(line)
            if s["round_idx"] != 2 or s["agent"] not in ("Teacher", "Adapter"):
                continue
            rows.append(
                {
                    "condition": s["condition"],
                    "task": s["task_id"],
                    "rep": s["repetition"],
                    "output": s["output"] or "",
                }
            )
    return pd.DataFrame(rows)


def episode_marker_table(df: pd.DataFrame) -> pd.DataFrame:
    def has_marker(g: pd.DataFrame) -> bool:
        task = g.name[0]
        joined = " ||| ".join(g["output"])
        return bool(MARKERS[task]["regex"].search(joined))

    ep = (
        df.groupby(["task", "condition", "rep"])
        .apply(has_marker, include_groups=False)
        .rename("marker")
        .reset_index()
    )
    return ep


def mcnemar_c4_vs_c0(ep: pd.DataFrame) -> tuple[int, int, float]:
    piv = ep.pivot_table(index=["task", "rep"], columns="condition", values="marker")
    b = int(((piv["C4"] == 1) & (piv["C0"] == 0)).sum())
    c = int(((piv["C4"] == 0) & (piv["C0"] == 1)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else float("nan")
    return b, c, p


def main() -> None:
    ep = episode_marker_table(load_round3())

    print("=== Marker definitions ===")
    for task, m in MARKERS.items():
        print(f"  {task:40s} {m['label']}")

    print("\n=== % of episodes (of 15) whose round 3 contains the marker ===")
    tab = (
        ep.pivot_table(index="task", columns="condition", values="marker", aggfunc="mean")
        * 100
    )
    print(tab[CONDITIONS].round(1).to_string())

    print("\n=== Pooled, six tasks (90 episodes per condition) ===")
    print((ep.groupby("condition")["marker"].mean() * 100).round(1).to_string())
    b, c, p = mcnemar_c4_vs_c0(ep)
    print(f"McNemar C4 vs C0: {b} vs {c}, p = {p:.2e}")

    print("\n=== Discriminative subset (C0 base rate < 15%) ===")
    eps = ep[ep["task"].isin(DISCRIMINATIVE)]
    print((eps.groupby("condition")["marker"].mean() * 100).round(1).to_string())
    b, c, p = mcnemar_c4_vs_c0(eps)
    print(f"McNemar C4 vs C0: {b} vs {c}, p = {p:.2e}")


if __name__ == "__main__":
    main()
