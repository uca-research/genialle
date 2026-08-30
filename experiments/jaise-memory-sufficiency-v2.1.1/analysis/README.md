# Post-hoc analyses over the published final-run logs

This folder contains the post-hoc analyses reported in the journal article
(JAISE submission), computed after the experimental run using only the
published logs in `../data/final_run/`. They require no re-execution of the
experiment and no GPU.

| Script | Reproduces | Input |
| --- | --- | --- |
| `mediation_analysis_v2.py` | Token-composition mechanism analysis (Section 5.6): whether the C4--C0 time and energy differences are statistically accounted for by completion-token composition. | `../data/final_run/episodes.jsonl.gz` |
| `continuity_check_v2.py` | Functional memory audit (Section 5.1): whether round-3 outputs reuse prior-round-specific information under longitudinal memory configurations. | `../data/final_run/steps.jsonl.gz` |

## Environment

These are **analysis-only** dependencies. They are intentionally *not* part of
the experimental runtime frozen in `../environment/requirements_frozen.txt`,
preserving the separation between the experimental platform and post-hoc
analysis.

```text
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

### `mediation_analysis_v2.py`

- Sanity checks reproduce the principal C4--C0 contrasts:
  approximately -1.356 s in execution time and -330.1 J in dynamic GPU energy.
- Dynamic energy: the C4 coefficient changes from approximately -330.1 J to
  -7.3 J (p = 0.57) after adding completion tokens as a covariate.
- Execution time: the corresponding coefficient changes from approximately
  -1.356 s to +0.035 s (p = 0.45).
- With prompt and completion tokens included simultaneously, no residual
  memory-configuration effect is statistically detectable
  (F-test p = 0.11 for execution time and p = 0.53 for dynamic GPU energy).
- Across the 90 matched trajectory-by-repetition blocks, C4--C0 differences in
  completion-token count correlate strongly with differences in dynamic GPU
  energy (Pearson r = 0.983) and execution time (Pearson r = 0.988).
  Spearman robustness checks give rho = 0.982 and rho = 0.988, respectively.

### `continuity_check_v2.py`

- Pooled across the six learner trajectories: 21.1% (C0) versus 78.9% (C4).
- In the discriminative subset (C0 base rate < 15%): 5.0% versus 88.3%;
  McNemar p = 1.8 x 10^-15.

## Frozen derived outputs

The following CSV files preserve the matched observations and correlation
results used in the token-composition mechanism analysis:

- `mechanism_paired_c4_c0.csv`: the 90 matched trajectory-by-repetition blocks,
  with C4--C0 differences in completion tokens, prompt tokens, dynamic GPU
  energy and execution time.
- `mechanism_correlations.csv`: Pearson and Spearman correlations between
  C4--C0 completion-token differences and the corresponding dynamic-energy
  and execution-time differences, including sample size and p-values.

These files are derived entirely from
`../data/final_run/episodes.jsonl.gz`; they contain no additional experimental
observations.

## Methodological note

Lexical markers were defined from the trajectory specifications in
`../config/tasks_v2.yaml` before inspecting output text. Two markers proved
topically natural and non-discriminative post hoc
(`overfitting_adaptation`, `llm_agents_study_plan`); the discriminative subset
(C0 base rate < 15%) is therefore a declared post-hoc sensitivity analysis.
The pooled six-trajectory analysis is reported together with this more
discriminative sensitivity analysis in the article.

The token-composition analysis is associative. The covariate models and
correlations are used to examine whether generation length is consistent with
the observed operational differences and are not interpreted as formal causal
mediation.
