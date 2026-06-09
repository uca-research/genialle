# Epistemic-Pedagogical Retrieval-Augmented Generation as a Service

This repository contains the implementation materials for the paper:

Epistemic-Pedagogical Retrieval-Augmented Generation as a Service for Educational Multi-Agent Systems.

## Contents

- app/: implementation of the retrieval, agent, telemetry, rendering, and minimum learning path components.
- experiments/: scripts, scenario configurations, evaluation questions, and experimental results.
- logs/: example query logs.
- .env.example: example local configuration file.

## Evaluated scenarios

The repository includes the materials for the three configurations evaluated in the paper:

1. Baseline retrieval-augmented generation.
2. Curated-pedagogical retrieval-augmented generation.
3. Curated-pedagogical retrieval-augmented generation with minimum learning path sequencing.

## Reproducing the summaries

The aggregated experimental summaries can be regenerated from the included JSONL result files with:

python experiments/summarize_experiment_mlp.py

## Models

Model weights are not redistributed. The local experiment used open pretrained models configured through environment variables. See .env.example.

## Data

The repository includes the evaluation question set and experimental logs. The full local vector index is not redistributed because it is a generated local artifact.

## Notes

This repository is intended to support transparency and reproducibility of the experimental workflow, including prompts, scenario configurations, evaluation questions, raw logs, and metric aggregation scripts.
