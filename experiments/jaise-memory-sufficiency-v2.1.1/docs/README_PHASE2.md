# IE26/JAISE experiment v2 - phase 2

This patch adds the complete corrected experimental runner while preserving the
original conference implementation in `app/`, `logs/`, and `experiments/`.

## What phase 2 adds

- Five memory conditions: C0-C4.
- Three-round longitudinal learner trajectories.
- Current-round isolation to remove duplicate context.
- Final-answer-only storage; reasoning is never stored.
- Block-randomized execution order.
- Matched seeds across memory conditions.
- Per-episode and per-call NVML energy integration.
- Raw compressed power traces.
- Robustness, memory-governance, and deletion metrics.
- Pilot acceptance analysis.

## Install

From the project root:

```bash
unzip -o IE26_JAISE_v2_phase2.zip -d .
bash scripts_v2/install_phase2.sh
python -m unittest discover -s tests_v2 -v
```

The llama.cpp server on `127.0.0.1:8033` must remain active.

## Smoke test

Runs five episodes: one task, one repetition, all five conditions.

```bash
bash scripts_v2/run_smoke_v2.sh
```

Then inspect the printed run directory and execute:

```bash
python scripts_v2/check_pilot_v2.py --run-dir logs_v2/<smoke_run>
```

The smoke test is a technical check; it is not part of the final dataset.

## Pilot

Runs 60 episodes: five conditions, six tasks, two repetitions.

```bash
bash scripts_v2/run_pilot_v2.sh
```

After completion:

```bash
python scripts_v2/check_pilot_v2.py --run-dir logs_v2/<pilot_run>
```

Do not run the full experiment until the pilot report passes or any failed
criterion has been reviewed.

## Full experiment

The configured full run contains 450 episodes and 4,050 LLM calls.

```bash
bash scripts_v2/run_full_v2.sh
```

## Resume an interrupted run

```bash
PYTHONPATH="$PWD" python -m app_v2.experiment_runner   --mode pilot   --run-dir logs_v2/<existing_run>   --resume
```
