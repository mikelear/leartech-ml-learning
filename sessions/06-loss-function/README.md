# Session 6: Loss Function + C++ Companion

## The Concept

Loss is a single number that measures **how wrong the prediction is**. The entire training process exists to make this number smaller.

- Prediction = 0.9, Truth = 1.0 → small loss (almost right)
- Prediction = 0.1, Truth = 1.0 → large loss (very wrong)

The loss function creates a **landscape** — like a hilly terrain. Training is walking downhill to find the lowest point (best weights). Gradients tell you which direction is downhill.

## Why a C++ Companion?

In Python, `loss.backward()` is a black box — gradients appear magically. In C++, you'll step through the computation:
- See the chain rule applied to each operation
- Watch gradients accumulate backwards through the graph
- Understand what "computation graph" actually means in memory

## Files

- `loss.py` — Python: loss functions, landscapes, gradient direction (PyCharm)
- `loss.cpp` — C++: manual backward pass, step through the chain rule (CLion)
- `CMakeLists.txt` — build config for C++ session

## Debugger Focus

### Python (PyCharm)

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 44 | Different predictions, same truth — see how loss changes |
| 2 | 66 | Loss landscape — loss values for every possible prediction |
| 3 | 89 | Gradient direction — which way is "downhill" |
| 4 | 115 | MSE vs BCE — two loss functions, different landscapes |
| 5 | 147 | Loss on our network — connecting back to Sessions 3-5 |
| 6 | 172 | Multiple examples — total loss across a batch |

### C++ (CLion)

| Breakpoint | Line | What to watch |
|-----------|------|---------------|
| 1 | 30 | Variables in memory — see the raw floats |
| 2 | 48 | Manual forward pass — multiply, add, sigmoid step by step |
| 3 | 64 | Manual loss computation — log and multiply |
| 4 | 82 | **THE KEY:** manual backward pass — chain rule step by step |
| 5 | 97 | The computed gradients — compare to PyTorch's automatic ones |
