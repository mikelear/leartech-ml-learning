# Session 7: Backpropagation Through Multiple Layers

## The Concept

In Session 6, you saw the chain rule for ONE neuron: three derivatives multiplied together. But a real network has multiple layers — how do gradients flow backwards through ALL of them?

The answer: the same chain rule, but applied layer by layer in reverse.

```
Forward:   Input → Layer1 → Layer2 → Layer3 → Loss
Backward:  Input ← Layer1 ← Layer2 ← Layer3 ← Loss
           (gradients flow BACKWARDS, each layer adds its link to the chain)
```

Each layer receives a gradient from the layer ahead of it (closer to the loss), multiplies by its own local derivative, and passes the result backwards to the layer behind it.

## Why This Matters

- **Layer 3** (closest to loss): gets the strongest, most direct gradient signal → learns fastest
- **Layer 1** (furthest from loss): gradient has been multiplied through 3 links → can be tiny → learns slowest
- This is the **vanishing gradient problem** — deep networks struggle because early layers get tiny gradients
- ReLU helps (gradient is either 0 or 1 — doesn't shrink), which is why it replaced sigmoid in hidden layers

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 62 | Computation graph — PyTorch's record of every operation |
| 2 | 80 | All gradients after backward — compare sizes across layers |
| 3 | 104 | Gradient flow visualised — which layers got strong vs weak signals |
| 4 | 130 | Vanishing gradient demo — sigmoid vs ReLU |
| 5 | 161 | Manual backprop — computing each layer's gradient by hand |
| 6 | 196 | Verifying manual matches PyTorch — proof we understand what backward() does |
| 7 | 226 | Gradient magnitude through training — do early layers learn? |
