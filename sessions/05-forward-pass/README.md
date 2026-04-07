# Session 5: Forward Pass Deep Dive

## The Concept

In Session 4 you saw data flow through layers: `[8] → [16] → [8] → [1]`. But what's actually happening at each step? This session zooms in.

A forward pass is the journey of data through the network — input in, prediction out. Every prediction the network ever makes (during training AND inference) is a forward pass.

**What happens at each layer:**
1. Every input gets multiplied by every weight (matrix multiply)
2. A bias is added
3. An activation function decides which neurons "fire"
4. The output becomes the input to the next layer

This session makes that visible — you'll watch specific numbers flow through, see how individual weights affect the output, and understand why the network makes the prediction it does.

## Debugger Focus

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 58 | The input features — your starting point |
| 2 | 71 | One neuron's computation: dot product + bias |
| 3 | 87 | All 16 neurons in layer 1 — see which ones fire (>0) and which die (→0) |
| 4 | 109 | Layer 2 — each neuron combines ALL of layer 1's outputs |
| 5 | 126 | The final neuron — how one number becomes the prediction |
| 6 | 157 | Trained network — same forward pass but now the numbers mean something |
| 7 | 183 | Neuron activation patterns — which neurons fire for bad vs good code |
| 8 | 211 | Killing neurons — what happens when you zero out specific weights |
