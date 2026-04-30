# Session 10.7: Fix Distribution Mismatch — Debug the Model

## Why this session exists

Session 10.6 blocked deployment: the improved model gets 57% eval accuracy
with 3 regressions. It predicts EVERYTHING as PASS (prob ~0.001).

This session teaches the **real ML workflow** — the debug loop:

```
train → eval → FAIL → diagnose → fix → retrain → eval → PASS
```

## The root cause

**Distribution mismatch.** Training data is large real PRs (median ~5,000 chars).
Eval test cases are short synthetic diffs (~300-600 chars). The TF-IDF vocabulary
learned from real PRs barely overlaps with eval diff tokens. The model sees
"nothing suspicious" and predicts PASS.

## Three fixes applied iteratively

| Fix | Problem addressed | Technique |
|-----|-------------------|-----------|
| 1. Char n-grams | Word vocabulary doesn't overlap | Character sequences ("eva","al(") are universal |
| 2. Boosted hand-crafted | 100 TF-IDF zeros drown 16 real signals | Multiply hand-crafted features by 3× |
| 3. Data augmentation | Model never saw short diffs | Add eval-like + variant diffs to training set |

## Running in PyCharm

1. Open `fix_distribution.py`
2. Set breakpoints at every `🔴 BREAKPOINT` line (7 breakpoints)
3. Debug and step through

### Key breakpoints

| # | What to inspect |
|---|----------------|
| 1 | TF-IDF token overlap: eval diffs activate almost no tokens |
| 2 | Char n-gram overlap: much better — characters are universal |
| 3 | Hand-crafted features firing on eval diffs (they catch eval, secrets!) |
| 4 | Augmented training set — eval-like variants added |
| 5 | Three model eval results side by side |
| 6 | Comparison table — which fix helped most? |
| 7 | Gate check — regressions + accuracy floor |

### What to try in the Evaluate window

```python
# How many TF-IDF tokens fire on eval diffs?
sum(tfidf_105.transform([test_cases[0]["diff_text"]]).toarray()[0] > 0)

# Same with char n-grams — should be much higher
sum(tfidf_char.transform([test_cases[0]["diff_text"]]).toarray()[0] > 0)

# Per-case results for each fix
for r in eval_f3['results']:
    print(f"{r['expected']:4s}→{r['actual']:4s} prob={r['prob']:.3f} {r['file']}")
```

## The learning

Every engineer knows this loop with code:
"Tests pass locally but fail in CI because the data is different."

This is the same thing with models:
"Model passes on training data but fails on eval because the distribution is different."

The fix is always the same: make your training environment match production.
