# Session 10: Our Classifier — PASS/FAIL on Real Code Review Data

## The Concept

Everything from Sessions 1-9 comes together. We build a REAL classifier trained on REAL feedback from the leartech AI code review pipeline.

**Input:** a code diff (from a PR)
**Output:** PASS or FAIL (probability)
**Training data:** 102 real feedback examples from `leartech-llm-training-data/feedback/`

## What Makes This Different from Sessions 3-9

- **Real data** — not hand-crafted examples. Real PRs, real diffs, real LLM verdicts.
- **Real features** — extracted from actual code patterns, not toy features.
- **Real evaluation** — precision, recall, F1 score, confusion matrix.
- **Exportable model** — save as `.pt` file, ready for deployment (Session 11).

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 72 | Real data loaded — inspect the diffs, labels, score distribution |
| 2 | 107 | Feature matrix — real features from real diffs |
| 3 | 130 | Train/val/test split — 70/15/15 |
| 4 | 158 | Model architecture — right-sized for our data |
| 5 | 193 | Training progress — loss and accuracy curves |
| 6 | 222 | Evaluation metrics — precision, recall, F1, confusion matrix |
| 7 | 258 | Predictions on test set — see which diffs it gets right/wrong |
| 8 | 282 | Save the model — ready for deployment |
