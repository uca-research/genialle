# IE26/JAISE experiment v2 - Phase 1

This patch creates an isolated v2 foundation inside the existing
`sma_edu_multi` repository. It does not overwrite the original experiment.

New components:

- `config/settings_v2.yaml`
- `config/tasks_v2.yaml`
- `app_v2/output_parser.py`
- `app_v2/llm_client.py`
- `app_v2/metrics/energy.py`
- `scripts_v2/preflight_v2.py`
- `tests_v2/`

Install from the repository root:

```bash
unzip IE26_JAISE_v2_phase1.zip -d .
bash scripts_v2/install_phase1.sh
python -m unittest discover -s tests_v2 -v
python scripts_v2/preflight_v2.py
```

Do not run the full experiment yet. The preflight must demonstrate:

1. The server is healthy.
2. The response contains clean final content.
3. Hidden reasoning is separated from final content.
4. The same request seed is accepted.
5. Prompt-cache reuse is disabled in the request.
6. NVML provides repeated power samples.
7. Every task contains exactly three learner-centred rounds.
