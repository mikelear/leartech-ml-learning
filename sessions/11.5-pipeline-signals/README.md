# Session 11.5: Pipeline Signals as Features — Multi-Modal Input

## Model affected: Our Classifier

This does NOT change Ollama, Claude, or DeepSeek. Those get leartech context via Layer 2 prompt injection (different mechanism). This session adds features to `extract_features()` in `leartech-ai-classifier/app/features.py`.

## Type: Both learning + production

Uses mock pipeline signals now. Features activate automatically when real infrastructure (risk-assessor, e2e task, semgrep rules) exists. Deploy today with defaults — improve later without code changes.

## The concept

Sessions 10–10.8 taught the classifier to read CODE (diffs). This session teaches it to read INFRASTRUCTURE (pipeline outputs).

```
Before:   diff text → [code features + TF-IDF] → PASS/FAIL
After:    diff text + pipeline JSON → [code + infra + TF-IDF] → PASS/FAIL
```

The diff tells you WHAT changed. Pipeline signals tell you WHAT THAT CHANGE MEANS in the system.

## The 6 new features

| Feature | Source | Available now? | Default |
|---|---|---|---|
| `services_affected` | risk-assessor AST analysis | No (mock) | 1 |
| `touches_critical` | service-catalog.yaml | No (mock) | 0 |
| `unexpected_edges` | Tempo span analysis | No (mock) | 0 |
| `coverage_gaps` | HAR vs known endpoints | No (mock) | 0 |
| `e2e_passed` | end2end/run.sh exit code | No (mock) | 1 |
| `leartech_violations` | semgrep leartech-* rules | No (mock) | 0 |

## Running in PyCharm

1. Open `pipeline_signals.py`
2. Set breakpoints at every `🔴 BREAKPOINT` line (7 breakpoints)
3. Debug and step through

### Key breakpoints

| # | What to inspect |
|---|----------------|
| 1 | `PIPELINE_FEATURES` — the 6 new infrastructure signals |
| 2 | Mock signals per eval test case — FAIL cases have violations, e2e failures |
| 3 | Combined feature vector `X_full` — 234 features wide |
| 4 | Model A vs Model B — with and without pipeline signals |
| 5 | Same diff, different signals — clean code but risky infra vs safe infra |
| 6 | Feature importance — which signals correlate most with FAIL |
| 7 | Production extractor — optional pipeline_signals with neutral defaults |

### The key moment (breakpoint 5)

Same diff. Same code features. Same TF-IDF. But different pipeline signals change the prediction. THIS is multi-modal input — the model sees context the diff alone can't provide.

## Production deployment

```python
# features.py — accepts optional pipeline signals
def extract_all_features(diff: str, pipeline_signals: dict | None = None):
    code = extract_features_v3(diff)           # 28 from diff
    infra = [                                    # 6 from pipeline
        pipeline_signals.get("services_affected", 1),
        pipeline_signals.get("touches_critical", 0),
        # ... defaults to neutral when not provided
    ] if pipeline_signals else [1, 0, 0, 0, 1, 0]
    combined = scaler.transform([code + infra]) * BOOST
    tfidf_feat = tfidf.transform([diff]).toarray()
    return np.hstack([combined, tfidf_feat])     # 234 total
```

Deploy with `pipeline_signals=None` today. When infra ships, pass real values — features activate without retraining.
