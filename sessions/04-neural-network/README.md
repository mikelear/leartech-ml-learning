# Session 4: A Neural Network — Stacking Neurons into Layers

## The Concept

In Session 3 you built ONE neuron: `dot(weights, features) + bias → sigmoid → prediction`

A neural network is **multiple neurons stacked into layers**, where the output of one layer becomes the input to the next:

```
Input [8 features]
    ↓
Layer 1: 16 neurons (each reads all 8 inputs → produces 1 output)
    = 16 outputs
    ↓
Layer 2: 8 neurons (each reads all 16 outputs from Layer 1)
    = 8 outputs
    ↓
Layer 3: 1 neuron (reads all 8 → produces final prediction)
    = 1 output (PASS/FAIL probability)
```

Each "layer" is just a matrix multiply — the same `W @ x + bias` from Session 1, but with a bigger W.

## Why Layers?

A single neuron can only learn straight-line boundaries ("if eval > 0, FAIL"). 
Multiple layers can learn COMBINATIONS: "eval + innerHTML together = definitely FAIL, but eval in a test file = probably OK". Each layer builds more abstract patterns from the previous layer's output.

## What You'll Build

A proper neural network using PyTorch's `nn.Module` — the standard way to build models. You'll see how it's just the Session 3 neuron repeated and stacked, but PyTorch handles the bookkeeping.

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 59 | Model parameters — weight matrices and biases for each layer |
| 2 | 84 | Layer 1 output — 8 features → 16 numbers (one per neuron in layer 1) |
| 3 | 95 | After ReLU — negative numbers zeroed out (the activation function) |
| 4 | 106 | Layer 2 output — 16 → 8 numbers |
| 5 | 114 | Final output — 8 → 1 number (the prediction) |
| 6 | 131 | All parameters — count them, these ARE the model |
| 7 | 163 | Training loss decreasing over steps |
| 8 | 197 | Predictions after training — the network learned |
