# Corrected experimental protocol

## Research framing

The educational multi-agent system is treated as a natural-language service
inside a human-centred intelligent learning environment. The environment
receives learner state, reasons through specialised agents, adapts its response
across time, and operates locally under resource and privacy constraints.

## Experimental conditions

The final experiment will compare:

- C0: no longitudinal memory.
- C1: bounded private memory per agent.
- C2: bounded shared blackboard.
- C3: bounded lexical retrieval memory.
- C4: data-minimising learner-state memory with role-based access.

C3 is explicitly lexical. C4 is the new journal extension: it stores structured
learner state but not full agent outputs or hidden reasoning.

## Human-centred interaction

Each episode contains three distinct learner events:

1. Initial request.
2. Feedback, changed constraint, or misconception.
3. Adaptation or transfer request.

The current event is explicitly available to every condition. Memory may
contain only earlier rounds. This removes the previous duplication of the
current Teacher and Adapter outputs.

## Experimental design

- Five conditions.
- Six tasks.
- Three rounds.
- Three agent calls per round.
- Two repetitions per cell in the pilot.
- Fifteen repetitions per cell in the full experiment.
- Randomised complete blocks: each repetition contains all condition-task
  combinations in a seeded random order.
- Matched inference seeds across conditions.
- Prompt-cache reuse disabled.
- Same model, hardware, prompts, role limits, and sampling parameters.

The full experiment contains 450 episodes and 4,050 LLM calls.

## Primary outcomes

1. Dynamic GPU energy per episode.
2. Prompt tokens per episode.
3. Blinded expert assessment of longitudinal adaptation and pedagogical
   usefulness, performed only after the computational pilot is stable.

## Secondary outcomes

- Total GPU energy.
- Wall-clock time and throughput.
- Per-agent and per-round energy and latency.
- Cached tokens.
- Truncation and empty-output rates.
- Retries and HTTP failures.
- Evaluator JSON parse failures.
- Memory context size, stored size, cross-agent exposure, and deletion success.
- Internal evaluator scores as diagnostics only.

## Energy protocol

Power is sampled repeatedly through NVML at 0.2-second intervals. Before every
episode, the system measures an idle baseline. Total GPU energy is obtained by
trapezoidal integration. Dynamic energy integrates the power above the median
idle baseline. Raw power samples are retained for audit.

## Responsibility safeguards

- Only final learner-facing content can enter memory.
- Hidden reasoning text is never stored.
- Current-round artifacts are excluded from memory reads.
- Memory is scoped to one episode and explicitly deleted.
- Synthetic learner profiles are used in the computational experiment.
- The internal Evaluator is not treated as independent pedagogical evidence.
- C4 uses data minimisation and role-based access.

## Pilot acceptance criteria

The full run must not start unless:

- All 60 pilot episodes complete.
- Non-empty final content is at least 99%.
- Stored outputs contain no raw reasoning markers.
- Evaluator JSON parse success is at least 99%.
- Truncation is below 2% for each agent.
- Every episode contains repeated power samples and a positive baseline.
- Memory-isolation tests pass.
