# Session 8: The Training Loop — Putting It All Together

## The Concept

You've now seen every piece:
- Features (Session 2): text → numbers
- Forward pass (Session 5): numbers → prediction
- Loss (Session 6): prediction vs truth → single error number
- Backward (Session 7): error → gradients for every weight

The **training loop** is just: repeat all four steps, many times, over all your data.

But there are important practical decisions:
- **Epochs**: how many times to loop over all the data?
- **Batches**: process examples one at a time, or in groups?
- **Learning rate**: how big a step to take each update?
- **Optimizers**: SGD, Adam, AdamW — different ways to use gradients
- **Learning rate scheduling**: start big, get smaller over time

This session is the most "production" session so far — the code looks like real training scripts.

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 71 | Training data as tensors — batched features and labels |
| 2 | 95 | One complete training step: forward → loss → backward → update |
| 3 | 121 | SGD vs Adam — different optimizers, different learning curves |
| 4 | 155 | Learning rate effect — too high (explodes), too low (stuck), just right |
| 5 | 191 | Learning rate scheduling — starts fast, slows down |
| 6 | 223 | Full training with validation — train loss vs val loss |
| 7 | 260 | The trained model — weights, predictions, what it learned |
