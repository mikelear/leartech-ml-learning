# Session 3: A Single Neuron

## The Concept

A neuron is the simplest possible prediction machine. It does ONE thing:

```
output = (weight₁ × input₁) + (weight₂ × input₂) + ... + bias
```

That's it. It's a **dot product** (from Session 1) plus a number (bias).

In code terms, if you have 4 input features:
```
output = weights[0]*features[0] + weights[1]*features[1] + weights[2]*features[2] + weights[3]*features[3] + bias
```

Or in tensor notation (from Session 1):
```
output = torch.dot(weights, features) + bias
```

## What Makes It "Learn"

The weights start as random numbers. The neuron makes terrible predictions.
Then we:
1. Compare its prediction to the right answer (loss)
2. Calculate how to adjust each weight to reduce the error (gradients)
3. Nudge each weight in the right direction

After many rounds of this, the weights settle on values that produce good predictions.

## This Session

You'll build a single neuron that predicts PASS/FAIL from the hand-crafted features of Session 2. You'll see:
- Random weights → bad prediction
- How the dot product works (the `@` from Session 1)
- What the activation function does (squishes output to 0-1)
- How `requires_grad` tracks everything for training

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 56 | Random weights — just numbers, no knowledge yet |
| 2 | 72 | Raw output of dot product — could be any number |
| 3 | 85 | After sigmoid — squished to 0-1 range (a probability) |
| 4 | 105 | Prediction vs truth — how wrong is it? |
| 5 | 122 | Gradients after backward — which weights need to change most |
| 6 | 142 | After one update — weights shifted, prediction improved |
| 7 | 167 | After 100 updates — the neuron has learned |
| 8 | 195 | Inspect final weights — which features matter most |

## Key Takeaway

A neural network is just layers of these neurons stacked together. If you understand one neuron, you understand the whole thing — it's just more of them.
