# Post-hoc analyses over the published final-run logs

This folder contains the two analyses reported in the journal article
(JAISE submission) that were computed **after** the experimental run,
using only the published logs in `../data/final_run/`. They require no
re-execution of the experiment and no GPU.

| Script | Reproduces | Input |
|---|---|---|
| `mediation_analysis_v2.py` | Mediation paragraph (Section 6.2): the C4–C0 time/energy advantage is accounted for by completion-token composition. | `data/final_run/episodes.jsonl.gz` |
| `continuity_check_v2.py` | Functional manipulation check (Section 5): round-3 outputs reference prior-round-specific content under memory conditions but not under C0. | `data/final_run/steps.jsonl.gz` |

## Environment

These are **analysis-only** dependencies. They are intentionally *not*
part of the experimental runtime frozen in
`../environment/requirements_frozen.txt`, preserving the separation
between the experimental platform and post-hoc analysis.

```
python >= 3.10
pandas
scipy
statsmodels   (mediation_analysis_v2.py only)
```

## Usage

```bash
cd analysis
python3 mediation_analysis_v2.py
python3 continuity_check_v2.py
```

## Expected key figures

`mediation_analysis_v2.py`
- Sanity check reproduces Table 7 means and Table 8 paired differences
  (C4−C0: −1.356 s, −330.1 J).
- Dynamic energy: C4 coefficient −330.1 J (p < 10⁻⁶) → −7.3 J (p = 0.57)
  after adding completion tokens; time: −1.356 s → +0.035 s (p = 0.45).
- With prompt and completion tokens included, no residual condition
  effect (F-test p = 0.11 energy, p = 0.53 time).
- Paired block-level correlations with Δcompletion tokens: r = 0.983
  (energy), r = 0.988 (time).

`continuity_check_v2.py`
- Pooled six tasks: 21.1% (C0) vs 78.9% (C4); McNemar 53 vs 1,
  p = 6.1×10⁻¹⁵.
- Discriminative subset (C0 base rate < 15%): 5.0% vs 88.3%;
  McNemar 50 vs 0, p = 1.8×10⁻¹⁵.

## Methodological note

Lexical markers were defined from the trajectory specifications in
`../config/tasks_v2.yaml` before inspecting output text. Two markers
proved topically natural and non-discriminative post hoc
(`overfitting_adaptation`, `llm_agents_study_plan`); the
"discriminative subset" (C0 base rate < 15%) is therefore a declared
post-hoc restriction, and the pooled six-task contrast is the primary
figure reported in the article.
