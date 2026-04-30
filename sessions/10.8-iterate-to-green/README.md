# Session 10.8: Iterate to Green — Make the Gate Pass

## The full debug loop

```
10.5: Train better model → 81% training accuracy
10.6: Eval harness BLOCKS → 57% eval, 3 regressions
10.7: Diagnose + fix → 71% eval, 1 regression
10.8: Targeted features + augmentation → 100% eval, GATE PASS ✅
```

## What changed

Two things fixed the remaining failures:

### 1. Targeted features (28 instead of 16)

The original 16 features count generic code patterns. The new 12 features
encode what a **reviewer** looks for — danger and quality signals:

| Feature | Type | Catches |
|---|---|---|
| `exec_system_pickle` | DANGER | `exec()`, `os.system`, `pickle.load` |
| `dangerous_html` | DANGER | `dangerouslySetInnerHTML`, `.innerHTML =` |
| `hardcoded_assignment` | DANGER | `API_KEY = 'sk-...'` |
| `untyped_python_def` | DANGER | `def foo(x):` without type annotations |
| `env_access` | DANGER | `os.environ`, `process.env` |
| `base64_decode` | DANGER | `base64.decode`, `atob()` |
| `type_annotations` | QUALITY | `: str`, `: int`, `: float` |
| `return_type` | QUALITY | `-> Response`, `-> dict` |
| `structured_types` | QUALITY | `BaseModel`, `dataclass`, `interface` |
| `test_assertions` | QUALITY | `assert`, `expect()`, `.should()` |
| `explicit_errors` | QUALITY | `fmt.Errorf`, `raise ValueError` |
| `async_typed` | QUALITY | `async def`, `Observable<>` |

### 2. Python-specific augmentation

The model confused "Python with eval" (FAIL) and "Python with pydantic" (PASS).
Added 6 Python variants (3 FAIL, 3 PASS) to teaching it the difference.

## Running in PyCharm

1. Open `iterate_to_green.py`
2. Set breakpoints at every `🔴 BREAKPOINT` line (6 breakpoints)
3. Debug and step through

### Key breakpoints

| # | What to inspect |
|---|----------------|
| 1 | `bad_python` vs `good_python` — the two diffs the model confuses |
| 2 | `bad_feats` vs `good_feats` — v3 features separate them clearly |
| 3 | Augmented dataset — targeted Python variants |
| 4 | Training accuracy — should be >85% |
| 5 | Eval results — per-case comparison with baseline |
| 6 | Gate decision — the payoff moment |

### What to try in the Evaluate window

```python
# Compare old vs new features on the problem cases
list(zip(FEATURE_NAMES_V3, bad_feats))
list(zip(FEATURE_NAMES_V3, good_feats))

# Check probability spread
[r['prob'] for r in results]
np.std([r['prob'] for r in results])
```

## Output files

| File | Purpose |
|---|---|
| `code_classifier_v4.pt` | Production-ready model weights |
| `tfidf_char_v4.pkl` | Character n-gram vectorizer |
| `scaler_v4.pkl` | Feature scaler (fitted on training data) |
| `new_baseline.json` | Updated baseline for eval suite |

## Production deployment

```
1. cp code_classifier_v4.pt ~/leartech/leartech-ai-classifier/models/
2. cp tfidf_char_v4.pkl ~/leartech/leartech-ai-classifier/models/
3. cp scaler_v4.pkl ~/leartech/leartech-ai-classifier/models/
4. Update features.py with extract_features_v3()
5. cp new_baseline.json ~/leartech/leartech-llm-training-data/evals/baseline.json
6. Open PR → eval pipeline confirms on both clusters → merge
```
