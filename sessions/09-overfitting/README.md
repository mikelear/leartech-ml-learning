# Session 9: Overfitting — When the Model Memorises Instead of Learning

## The Concept

A model that gets 100% on training data but fails on new data has **overfit**. It memorised the specific training examples instead of learning general patterns.

Think of it like studying for an exam:
- **Learning:** understanding the concepts → can answer any question
- **Memorising:** remembering specific answers → fails on new questions

## How to Detect It
- **Training loss keeps decreasing**, validation loss starts INCREASING
- Gap between train accuracy (high) and val accuracy (lower) keeps growing

## How to Fix It
1. **More training data** — the more examples, the harder to memorise
2. **Dropout** — randomly disable neurons during training, forces redundancy
3. **Early stopping** — stop training when val_loss stops improving
4. **Regularisation** — penalise large weights (L2/weight decay)
5. **Simpler model** — fewer parameters = less capacity to memorise

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 78 | Training a model that WILL overfit — too many parameters, too little data |
| 2 | 111 | Train vs val loss curves — watch them diverge (the overfit signature) |
| 3 | 143 | Dropout effect — same network with and without dropout |
| 4 | 172 | Early stopping — automatically stop when val_loss stops improving |
| 5 | 214 | Weight decay (L2 regularisation) — penalising large weights |
| 6 | 241 | All techniques compared side by side |
