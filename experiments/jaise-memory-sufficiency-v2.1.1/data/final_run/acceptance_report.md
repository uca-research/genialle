# Pilot acceptance report

- Run: `logs_v2/full_20260711_212350`
- Mode: `full`
- Completed episodes: 450/450
- Logged calls: 4050
- Failures: 0
- Overall result: **PASS**

| Check | Result | Detail |
|---|---|---|
| All scheduled episodes completed | PASS | completed=450, expected=450, failures=0 |
| Non-empty final content >= 99% | PASS | empty=0/4050 (0.00%) |
| No raw reasoning markers stored | PASS | marker_outputs=0 |
| Evaluator JSON validity >= 99% | PASS | valid=1350/1350 (100.00%) |
| Adapter truncation < 2% | PASS | truncated=2/1350 (0.15%) |
| Evaluator truncation < 2% | PASS | truncated=0/1350 (0.00%) |
| Teacher truncation < 2% | PASS | truncated=3/1350 (0.22%) |
| Valid power traces | PASS | insufficient_samples=0, nonpositive_baseline=0 |
| No current-round memory contamination | PASS | contaminated_reads=0 |
| Episode memory deletion succeeded | PASS | deletion_failures=0 |
