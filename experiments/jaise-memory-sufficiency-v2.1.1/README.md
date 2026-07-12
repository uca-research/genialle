# Memory sufficiency in an educational multi-agent system

This directory contains the code, frozen configuration, experimental
records and derived results associated with the study:

**Memory sufficiency for responsible educational multi-agent systems:
A controlled study of energy, robustness and data minimisation**

## Experimental design

- Five memory conditions: C0-C4
- Six longitudinal educational tasks
- Fifteen repetitions
- 450 episodes
- 4,050 model calls
- Three agents: Teacher, Adapter and Evaluator
- One locally served open-source LLM
- One NVIDIA GeForce RTX 4080 SUPER GPU

## Main folders

- `src/app_v2/`: corrected experimental implementation
- `config/`: frozen experiment and task configuration
- `scripts/`: execution and acceptance scripts
- `tests/`: unit tests
- `docs/`: experimental protocol and implementation notes
- `data/final_run/`: final episode-level and call-level records
- `results/`: derived statistical tables, when available
- `environment/`: runtime and reproducibility metadata

## Final execution

The accepted final execution completed:

- 450 of 450 episodes
- 4,050 of 4,050 model calls
- 0 failed episodes
- 0 empty outputs
- 5 truncated calls, representing 0.12% of all calls
- 1,350 of 1,350 syntactically valid evaluator JSON records
- 0 current-round memory contamination events
- 0 memory deletion failures

## Important scope limitation

The repository supports claims about computational efficiency,
robustness, memory governance and data minimisation.
