# Post-hoc analyses over the published final-run logs

This folder contains the post-hoc analyses reported in the journal article
(JAISE submission), together with frozen derived outputs used to document the
reported results. All analyses use only the published final-run logs in
`../data/final_run/`. They require no re-execution of the experiment and no GPU.

| File | Purpose | Input / origin |
| --- | --- | --- |
| `mediation_analysis_v2.py` | Token-composition mechanism analysis (Section 5.6): examines whether the C4--C0 time and dynamic-energy differences are statistically accounted for by completion-token composition. | `../data/final_run/episodes.jsonl.gz` |
| `continuity_check_v2.py` | Functional memory audit (Section 5.1): examines whether round-3 Teacher and Adapter outputs reuse prior-round-specific information under longitudinal memory configurations. | `../data/final_run/steps.jsonl.gz` |
| `mechanism_paired_c4_c0.csv` | Frozen matched-block dataset for the token-composition analysis: 90 trajectory-by-repetition C4--C0 differences in completion tokens, prompt tokens, dynamic GPU energy and execution time. | Derived from `../data/final_run/episodes.jsonl.gz` |
| `mechanism_correlations.csv` | Frozen Pearson and Spearman correlation results for the token-composition mechanism analysis, including sample size and p-values. | Derived from `mechanism_paired_c4_c0.csv` |

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
- After adding completion tokens as a covariate, the C4--C0 dynamic-energy
  coefficient is attenuated from approximately -330.1 J to -7.3 J
  (`p = 0.57`).
- The corresponding execution-time coefficient changes from approximately
  -1.356 s to +0.035 s (`p = 0.45`).
- When prompt and completion tokens are included simultaneously, no residual
  memory-configuration effect is statistically detectable for execution time
  (`F`-test, `p = 0.11`) or dynamic GPU energy
  (`F`-test, `p = 0.53`).
- Across the 90 matched trajectory-by-repetition blocks, C4--C0 differences in
  completion-token count are strongly associated with the corresponding
  differences in dynamic GPU energy and execution time.

### `mechanism_paired_c4_c0.csv`

The file contains one row for each of the 90 matched
trajectory-by-repetition blocks and the following C4--C0 differences:

- completion-token count;
- prompt-token count;
- dynamic GPU energy;
- execution time.

Mean C4--C0 differences provide an additional sanity check:

- completion tokens: approximately -228.12;
- prompt tokens: approximately +97.86;
- dynamic GPU energy: approximately -330.13 J;
- execution time: approximately -1.356 s.

### `mechanism_correlations.csv`

The frozen correlation results are:

- completion-token difference vs dynamic GPU energy difference:
  Pearson `r = 0.983`, `p = 1.09 x 10^-66`;
- completion-token difference vs execution-time difference:
  Pearson `r = 0.988`, `p = 1.80 x 10^-72`;
- completion-token difference vs dynamic GPU energy difference:
  Spearman `rho = 0.982`;
- completion-token difference vs execution-time difference:
  Spearman `rho = 0.988`.

These values correspond to the token-composition mechanism analysis reported
in Section 5.6 of the manuscript.

### `continuity_check_v2.py`

- Pooled across the six learner trajectories, prior-round lexical markers are
  detected in 21.1% of C0 episodes and 78.9% of C4 episodes.
- The corresponding pooled matched comparison gives McNemar discordant counts
  of 53 versus 1 (`p = 6.1 x 10^-15`).
- In the more discriminative subset, defined by a C0 marker base rate below
  15%, marker presence is 5.0% under C0 and 88.3% under C4.
- The matched comparison in this subset gives McNemar discordant counts of
  50 versus 0 (`p = 1.8 x 10^-15`).

## Frozen derived outputs

The two CSV files preserve the matched observations and correlation results
used to document the token-composition mechanism analysis:

- `mechanism_paired_c4_c0.csv` contains the 90 matched
  trajectory-by-repetition blocks used for the C4--C0 comparison.
- `mechanism_correlations.csv` contains the Pearson and Spearman coefficients,
  sample sizes and p-values calculated from those matched differences.

Both files are derived entirely from
`../data/final_run/episodes.jsonl.gz`. They contain no additional experimental
observations and do not alter the original final-run data.

## Methodological note

The token-composition analysis is exploratory and associative. The covariate
models and matched-block correlations examine whether generation length is
consistent with the observed operational differences. They are not interpreted
as formal causal mediation and do not establish that learner-state memory
directly causes lower energy consumption.

For the functional memory audit, lexical markers were defined from the learner
trajectory specifications in `../config/tasks_v2.yaml` before inspecting the
generated output text. Two trajectories,
`overfitting_adaptation` and `llm_agents_study_plan`, contained markers that
could also arise naturally from the current topic under the stateless baseline.
The restriction to trajectories with a C0 marker prevalence below 15% is
therefore reported as a post-hoc sensitivity analysis. Results for all six
learner trajectories are retained alongside this more discriminative subset.

## Reproducibility scope

The files in this folder reproduce or preserve analyses conducted after the
experimental run. The original experimental observations remain frozen in
`../data/final_run/`. No model weights, hidden reasoning traces, student data or
additional experimental observations are introduced by these analyses.
````
