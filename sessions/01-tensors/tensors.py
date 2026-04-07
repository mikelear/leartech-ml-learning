"""
Session 1: What is a Tensor?

Open this file in PyCharm. Set breakpoints at every line marked with 🔴 BREAKPOINT.
Run in Debug mode (right-click → Debug).

At each breakpoint, inspect the variables in the Variables pane.
Hover over tensors to see their values. Use the Evaluate Expression dialog (Alt+F8)
to try things like: scalar.shape, vector.dtype, matrix[0], etc.
"""

import torch
import numpy as np

# ============================================================
# PART 1: Creating tensors
# ============================================================

# A scalar — just a single number, but wrapped in a tensor
scalar = torch.tensor(42.0)
print(f"Scalar: {scalar}")
print(f"  value: {scalar.item()}")  # .item() extracts the Python number
print(f"  shape: {scalar.shape}")   # torch.Size([]) — zero dimensions
print(f"  dtype: {scalar.dtype}")   # float32
print(f"  ndim:  {scalar.ndim}")    # 0 — it's a 0-dimensional tensor
x = 1  # 🔴 BREAKPOINT — inspect scalar in Variables pane
# Notice: scalar.shape is torch.Size([]) — empty! It has no dimensions.
# Try in Evaluate: scalar + 10, scalar * 2

# A vector — a 1D list of numbers
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"\nVector: {vector}")
print(f"  shape: {vector.shape}")   # torch.Size([3]) — one dimension, 3 elements
print(f"  dtype: {vector.dtype}")
print(f"  ndim:  {vector.ndim}")    # 1
x = 2  # 🔴 BREAKPOINT — inspect vector
# Notice: shape is [3]. One dimension with 3 elements.
# Try in Evaluate: vector[0], vector[-1], vector.sum(), vector.mean()

# A matrix — a 2D grid of numbers
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print(f"\nMatrix: {matrix}")
print(f"  shape: {matrix.shape}")   # torch.Size([2, 3]) — 2 rows, 3 columns
print(f"  ndim:  {matrix.ndim}")    # 2
x = 3  # 🔴 BREAKPOINT — inspect matrix
# Notice: shape is [2, 3]. Read it as "2 rows, 3 columns".
# Try in Evaluate: matrix[0] (first row), matrix[1, 2] (row 1, col 2 = 6.0)
# Try: matrix.T (transpose — swaps rows and columns, shape becomes [3, 2])

# A 3D tensor — think of it as a stack of matrices
tensor_3d = torch.tensor([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
], dtype=torch.float32)
print(f"\n3D Tensor: shape = {tensor_3d.shape}")  # [2, 3, 4]
print(f"  ndim: {tensor_3d.ndim}")  # 3
x = 4  # 🔴 BREAKPOINT — inspect tensor_3d
# Shape is [2, 3, 4]: 2 matrices, each has 3 rows and 4 columns.
# Try in Evaluate: tensor_3d[0] (first matrix), tensor_3d[1, 2, 3] (= 24.0)
# Try: tensor_3d.shape[0], tensor_3d.shape[1], tensor_3d.shape[2]


# ============================================================
# PART 2: NumPy ↔ PyTorch (they share memory!)
# ============================================================

# Create a NumPy array
np_array = np.array([10.0, 20.0, 30.0])

# Convert to PyTorch tensor — SHARES the same memory
torch_from_np = torch.from_numpy(np_array)

print(f"\nNumPy array: {np_array}")
print(f"Torch tensor: {torch_from_np}")
x = 5  # 🔴 BREAKPOINT — inspect both np_array and torch_from_np

# Modify the NumPy array
np_array[0] = 999.0
print(f"\nAfter modifying NumPy:")
print(f"  NumPy:  {np_array}")
print(f"  Torch:  {torch_from_np}")  # Also changed! They share memory.
x = 6  # 🔴 BREAKPOINT — both changed! This is important to understand.
# The tensor and array point to the SAME memory. Changing one changes the other.
# This is efficient (no copy) but can cause bugs if you're not aware.


# ============================================================
# PART 3: Operations — tensors do maths
# ============================================================

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

# Element-wise operations (each element independently)
added = a + b           # [11, 22, 33]
multiplied = a * b      # [10, 40, 90]
squared = a ** 2        # [1, 4, 9]
x = 7  # 🔴 BREAKPOINT — inspect added, multiplied, squared
# These operations work element-by-element. No loops needed.
# This is why ML is fast — operations on entire arrays at once.

# Dot product — sum of element-wise multiplication
# This is THE fundamental operation in neural networks
dot_product = torch.dot(a, b)  # 1*10 + 2*20 + 3*30 = 140
print(f"\nDot product: {dot_product}")
x = 8  # 🔴 BREAKPOINT — inspect dot_product
# dot_product is a scalar (single number).
# A neuron computes: dot_product(weights, inputs) + bias
# This is literally what a neural network does, billions of times.

# Matrix multiplication — the heart of deep learning
W = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])  # Shape: [3, 2]
x_input = torch.tensor([0.5, 0.7])  # Shape: [2]

result = W @ x_input  # Matrix multiply: [3, 2] @ [2] = [3]
# Row 0: 1*0.5 + 2*0.7 = 1.9
# Row 1: 3*0.5 + 4*0.7 = 4.3
# Row 2: 5*0.5 + 6*0.7 = 6.7
print(f"\nMatrix multiply: {result}")
print(f"  W shape: {W.shape}")
print(f"  x shape: {x_input.shape}")
print(f"  result shape: {result.shape}")
x = 9  # 🔴 BREAKPOINT — inspect W, x_input, result
# The shapes tell you everything:
#   [3, 2] @ [2] = [3]
# The inner dimensions must match (2 == 2).
# The result takes the outer dimension (3).
# This is how a neural network layer transforms input → output.


# ============================================================
# PART 4: requires_grad — tracking operations for training
# ============================================================

# A normal tensor — no tracking
normal = torch.tensor([1.0, 2.0, 3.0])
print(f"\nNormal tensor requires_grad: {normal.requires_grad}")  # False

# A tensor that tracks operations — needed for training
tracked = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Tracked tensor requires_grad: {tracked.requires_grad}")  # True

# Do some operations on the tracked tensor
y = tracked * 2        # PyTorch remembers: "y = tracked * 2"
z = y.sum()            # PyTorch remembers: "z = sum(y)"

x = 10  # 🔴 BREAKPOINT — inspect tracked, y, z
# Look at z.grad_fn — it shows the operation that created z (SumBackward0)
# Look at y.grad_fn — it shows MulBackward0
# This is the "computation graph" — PyTorch recorded every operation.
# In Session 7 (backpropagation), we'll call z.backward() and
# PyTorch will walk backwards through this graph to compute gradients.

# Compute gradients (we'll understand this fully in Session 7)
z.backward()
print(f"\nGradients of tracked: {tracked.grad}")
# The gradient tells us: "how much does z change when we change each element of tracked?"
# Since z = sum(tracked * 2), the gradient is [2, 2, 2] — each element contributes 2.
x = 11  # 🔴 BREAKPOINT — inspect tracked.grad
# This is a preview of backpropagation. Don't worry if the maths
# isn't clear yet — Session 7 will make it concrete.


# ============================================================
# PART 5: Reshaping — changing the shape without changing the data
# ============================================================

original = torch.arange(12, dtype=torch.float32)  # [0, 1, 2, ..., 11]
print(f"\nOriginal: {original}")
print(f"  shape: {original.shape}")  # [12]

# Same 12 numbers, different shapes
as_matrix = original.reshape(3, 4)     # 3 rows, 4 columns
as_matrix2 = original.reshape(4, 3)    # 4 rows, 3 columns
as_3d = original.reshape(2, 2, 3)      # 2 blocks of 2 rows × 3 columns

x = 12  # 🔴 BREAKPOINT — inspect original, as_matrix, as_matrix2, as_3d
# The DATA is the same 12 numbers [0..11]. Only the SHAPE changed.
# as_matrix: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
# as_matrix2: [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
# 3 * 4 = 4 * 3 = 2 * 2 * 3 = 12 — total elements must be the same.

print(f"\nReshaped to [3, 4]:\n{as_matrix}")
print(f"\nReshaped to [4, 3]:\n{as_matrix2}")
print(f"\nReshaped to [2, 2, 3]:\n{as_3d}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("Session 1 Complete!")
print("=" * 50)
print("""
Key concepts:
1. Tensors are containers for numbers with a SHAPE
2. Shape tells you the dimensions: [2, 3] = 2 rows, 3 columns
3. Operations work element-wise (no loops needed)
4. Matrix multiply (@) is the core operation in neural networks
5. requires_grad=True tells PyTorch to track operations (for training)
6. Reshape changes shape but not data

Next session: Feature Extraction — turning code diffs into numbers
""")
