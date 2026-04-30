# Session 10.6: Eval Harness — Unit Testing for Models

## The analogy

| Code testing | Model eval |
|---|---|
| `test_login.py` | `test_cases/fail/hardcoded-secrets.diff` |
| `assert status == 200` | `assert verdict == "FAIL"` |
| coverage >= 80% | accuracy >= 80% |
| no test failures | no regressions vs baseline |
| `pytest --snapshot-update` | update `baseline.json` |
| CI blocks merge | pipeline blocks model deploy |

## Prerequisites

- Session 10.5 completed (model + artefacts saved in `10.5-better-features/`)
- `pip install pyyaml` (if not already installed)

## Running in PyCharm

1. Open `eval_harness.py`
2. Set breakpoints at every `🔴 BREAKPOINT` line (6 breakpoints)
3. Debug and step through

### Key breakpoints

| # | What to inspect |
|---|----------------|
| 1 | `test_cases` — the test suite (like pytest fixtures) |
| 2 | `baseline` — current production model results (43% accuracy) |
| 3 | `predict()` results — sanity check new model on obvious inputs |
| 4 | `results` — full eval run, compare probabilities to baseline |
| 5 | `improved`, `regressed`, `unchanged` — regression detection |
| 6 | `gate_passed`, `gate_reasons` — the deploy/no-deploy decision |

### What to try in the Evaluate window

```python
# Probability comparison
[b['probability'] for b in baseline]   # all ~0.5
[r['probability'] for r in results]    # hopefully spread out

# How many cases improved?
len(improved)

# Check a specific regression
regressed[0] if regressed else "no regressions"
```

## Production mapping

This session's code maps directly to `leartech-llm-training-data/evals/run_evals.py`:

| This session | Production |
|---|---|
| `manifest.yaml` loaded → test_cases | Same manifest, same test cases |
| `predict(diff)` using Session 10.5 model | `run_eval_with_model()` loads from `leartech-ai-classifier` |
| `baseline.json` comparison | Same baseline, same comparison logic |
| Gate decision (accuracy + regressions) | Currently exits 0 always — needs the gate logic from this session |

## Output

- `new_baseline.json` — if gate passes, this becomes the new baseline
- To update production: `cp new_baseline.json ~/leartech/leartech-llm-training-data/evals/baseline.json`
