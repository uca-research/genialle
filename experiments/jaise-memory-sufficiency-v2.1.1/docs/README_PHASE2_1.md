# IE26/JAISE experiment v2 - phase 2.1

This corrective patch addresses the smoke-test truncations.

## Root cause

The GPT-OSS Harmony template uses the request field
`chat_template_kwargs.reasoning_effort`. The earlier client sent
`enable_thinking`, so GPT-OSS retained its default medium reasoning effort.

## Changes

- Send `reasoning_effort` explicitly for every role.
- Use `low` reasoning effort for Teacher, Adapter, and Evaluator.
- Raise generation ceilings without changing output-length instructions:
  - Teacher: 1536 tokens
  - Adapter: 1280 tokens
  - Evaluator: 640 tokens
- Add a unit test that verifies the actual request payload.

## Install

```bash
unzip -o IE26_JAISE_v2_phase2_1.zip -d .
bash scripts_v2/install_phase2.sh
python -m unittest discover -s tests_v2 -v
```

Then run a new smoke test. Do not reuse or append to the previous run:

```bash
bash scripts_v2/run_smoke_v2.sh
```
