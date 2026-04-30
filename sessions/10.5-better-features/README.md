# Session 10.5: Better Features — TF-IDF + Feature Stacking

## Why this session exists

Session 10's classifier has 43% accuracy on the eval suite. All PASS test cases
get probabilities between 0.52–0.57 — the model is uncertain and defaults to FAIL.

The root cause: 16 hand-crafted regex features where most are zero (eval_calls,
innerHTML, secret_names are ~0 for almost every diff). The model has almost nothing
to learn from.

## What you'll learn

1. **TF-IDF** — automatic feature discovery from text. Finds distinctive words
   without you prescribing them. This is how search engines work.
2. **Feature stacking** — combining hand-crafted domain knowledge with automatic
   discovery. Your regex features + TF-IDF features = better together.
3. **Feature scaling** — why StandardScaler is mandatory when combining features
   of different scales.
4. **Probability spread** — why a confident model matters more than raw accuracy.

## Setup

```bash
cd ~/leartech/ml-learning
source .venv/bin/activate
pip install scikit-learn  # if not already installed
```

## Running in PyCharm

1. Open `better_features.py` in PyCharm
2. Set breakpoints at every `🔴 BREAKPOINT` line (7 breakpoints)
3. Debug (not Run) — step through and inspect variables

### Key breakpoints

| # | Line | What to inspect |
|---|------|----------------|
| 1 | 62 | `raw_data` — the real feedback records, class balance |
| 2 | 116 | `stds` — which hand-crafted features actually vary (most don't) |
| 3 | 176 | `features_tfidf`, `feature_names_tfidf` — what TF-IDF found |
| 4 | 209 | `features_stacked` — combined [116] feature vector |
| 5 | 311 | `result_a/b/c` — compare three models, probability distributions |
| 6 | 347 | `results` — head-to-head comparison table |
| 7 | 393 | Saved artefacts — model + vectorizer + scaler |

### What to try in the Evaluate window

```python
# How many features have near-zero variance?
(stds < 0.01).sum()

# Top TF-IDF tokens for FAIL diffs
feature_names_tfidf[diff_weights.argsort()[::-1][:5]]

# Probability histogram for baseline model
np.histogram(result_a['probabilities'], bins=10)

# Same for stacked model — should be more spread out
np.histogram(result_c['probabilities'], bins=10)
```

## Output files

| File | Purpose | Prod equivalent |
|------|---------|----------------|
| `code_classifier_v2.pt` | Trained model weights | `leartech-ai-classifier/models/code_classifier.pt` |
| `tfidf_vectorizer.pkl` | TF-IDF vocabulary + IDF weights | New artefact for `leartech-ai-classifier/models/` |
| `feature_scaler.pkl` | StandardScaler fit on training data | New artefact for `leartech-ai-classifier/models/` |

## Production mapping

This session's code maps to `leartech-ai-classifier/app/features.py`:

```python
# Current (Session 10)
def extract_features(diff: str) -> torch.Tensor:
    # 16 regex counts → [16]

# After this session (Session 10.5)
def extract_features(diff: str) -> torch.Tensor:
    # 16 regex counts → [16]

def tfidf_features(diff: str, vectorizer) -> np.ndarray:
    # TF-IDF transform → [100]

def stacked_features(diff: str, vectorizer, scaler) -> torch.Tensor:
    # hand-crafted (scaled) + TF-IDF → [116]
```

## Next: Session 10.6 — Eval Harness

Run the improved model against `leartech-llm-training-data/evals/test_cases/` and
compare to `baseline.json`. Gate: don't deploy if accuracy < 80%.
