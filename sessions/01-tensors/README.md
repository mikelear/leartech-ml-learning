# Session 1: What is a Tensor?

## The Concept

Everything in ML is numbers. A tensor is just a container for numbers — like an array, but with superpowers:

- A **scalar** is a single number: `42`
- A **vector** is a 1D array: `[1, 2, 3]`
- A **matrix** is a 2D array: `[[1, 2], [3, 4]]`
- A **tensor** is the general term for any of these (including 3D, 4D, etc.)

PyTorch tensors are like NumPy arrays but they can:
1. Track operations for automatic differentiation (backpropagation)
2. Run on GPUs
3. Be part of a computation graph

In this session you'll see what tensors look like in memory, how they have shape and dtype, and how operations on them work.

## Files

- `tensors.py` — the main exercise. Open in PyCharm.

## PyCharm Setup

1. Open the `ml-learning` folder as a project in PyCharm
2. Set the Python interpreter to `.venv/bin/python`:
   - PyCharm → Settings → Project → Python Interpreter → Add → Existing → `.venv/bin/python`
3. Open `sessions/01-tensors/tensors.py`
4. Set breakpoints where you see `# 🔴 BREAKPOINT` comments
5. Right-click → Debug

## What to Watch in the Debugger

At each breakpoint, look at the **Variables** pane:

| Variable | What to notice |
|----------|---------------|
| `scalar` | It's a tensor with 0 dimensions (just a number wrapped in a tensor) |
| `vector` | Shape is `[3]` — one dimension with 3 elements |
| `matrix` | Shape is `[2, 3]` — 2 rows, 3 columns |
| `tensor_3d` | Shape is `[2, 3, 4]` — think of it as 2 matrices, each 2×3 |
| `.dtype` | The number type — float32 is standard for ML |
| `.shape` | The dimensions — this is THE most important property |
| `.requires_grad` | Does PyTorch track operations on this? (needed for training) |

## Key Takeaway

**Shape is everything.** When something goes wrong in ML, 90% of the time it's a shape mismatch. Get comfortable reading shapes.
