"""
Session 4: A Neural Network — Stacking Neurons into Layers

Session 3 was one neuron: dot(weights, features) + bias
This session: multiple neurons organised into layers.

A layer of 16 neurons is just a matrix multiply:
    W[16, 8] @ input[8] + bias[16] = output[16]
    (16 neurons, each with 8 weights = a [16, 8] weight matrix)

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import re


# ============================================================
# Features from Session 2
# ============================================================

def extract_features(diff: str) -> torch.Tensor:
    features = [
        len(re.findall(r'eval\s*\(', diff)),
        len(re.findall(r'innerHTML', diff)),
        len(re.findall(r'(API_KEY|SECRET|PASSWORD)', diff, re.IGNORECASE)),
        len(re.findall(r'(sk-|ghp_|password)', diff, re.IGNORECASE)),
        len(re.findall(r'import\s+', diff)),
        len(re.findall(r'constructor', diff)),
        len(re.findall(r'(subscribe|Observable)', diff)),
        len(re.findall(r'(HttpClient|DomSanitizer)', diff)),
    ]
    return torch.tensor(features, dtype=torch.float32)


# ============================================================
# PART 1: Build a neural network with PyTorch
# ============================================================

class CodeReviewNet(nn.Module):
    """
    A 3-layer neural network for PASS/FAIL prediction.

    nn.Module is PyTorch's base class for all models.
    It handles: parameter tracking, saving/loading, GPU transfer.
    """

    def __init__(self):
        super().__init__()
        # Layer 1: 8 inputs → 16 neurons
        # This creates a [16, 8] weight matrix and [16] bias vector
        self.layer1 = nn.Linear(8, 16)

        # Layer 2: 16 inputs → 8 neurons
        self.layer2 = nn.Linear(16, 8)

        # Layer 3 (output): 8 inputs → 1 neuron (PASS/FAIL)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        """
        The forward pass — data flows through the network.
        This is called when you do: model(input)
        """
        # Each layer: matrix multiply → activation function
        x = self.layer1(x)      # [8] → [16]  (W @ input + bias)
        x = torch.relu(x)       # Zero out negatives (activation)
        x = self.layer2(x)      # [16] → [8]
        x = torch.relu(x)       # Activation again
        x = self.layer3(x)      # [8] → [1]
        x = torch.sigmoid(x)    # Squish to 0-1 probability
        return x


# Create the model
torch.manual_seed(42)
model = CodeReviewNet()

print("=" * 60)
print("PART 1: Network architecture")
print("=" * 60)
print(model)
x = 1  # 🔴 BREAKPOINT — Line 76: inspect model
# In the Variables pane, expand 'model':
#   model.layer1.weight → shape [16, 8] (16 neurons × 8 inputs each)
#   model.layer1.bias   → shape [16]    (one bias per neuron)
#   model.layer2.weight → shape [8, 16] (8 neurons × 16 inputs)
#   model.layer3.weight → shape [1, 8]  (1 neuron × 8 inputs)
#
# These weight matrices ARE the model. Random numbers now, knowledge later.
# Try in Evaluate: model.layer1.weight.shape, model.layer1.weight


# ============================================================
# PART 2: Step through the forward pass
# ============================================================

bad_diff = """
+ const API_KEY = 'sk-proj-abc123';
+ const parsed = eval('(' + input + ')');
+ document.getElementById('output').innerHTML = parsed.html;
"""

features = extract_features(bad_diff)
print(f"\nInput features: {features.tolist()}")
print(f"Input shape: {features.shape}")  # [8]

# Step through each layer manually to see what happens
print(f"\n{'='*60}")
print("PART 2: Step through the forward pass")
print(f"{'='*60}")

# Layer 1: 8 features → 16 hidden values
hidden1_raw = model.layer1(features)  # W[16,8] @ features[8] + bias[16] = [16]
print(f"\nLayer 1 raw output: shape {hidden1_raw.shape}")
print(f"  Values: {hidden1_raw.data[:6].tolist()}")  # First 6 values
x = 2  # 🔴 BREAKPOINT — Line 101: inspect hidden1_raw
# Shape [16] — 16 numbers, one from each neuron in layer 1.
# Some positive, some negative. This is the raw dot product + bias.
# Try in Evaluate: hidden1_raw.shape, hidden1_raw.min(), hidden1_raw.max()

# ReLU activation: replace all negatives with 0
# Why? Without activation, stacking layers would just be one big matrix multiply.
# ReLU adds non-linearity — the network can learn curves, not just straight lines.
hidden1 = torch.relu(hidden1_raw)
print(f"\nAfter ReLU: {hidden1.data[:6].tolist()}")
negatives_removed = (hidden1_raw < 0).sum().item()
print(f"  ReLU zeroed out {negatives_removed} of {hidden1.shape[0]} values")
x = 3  # 🔴 BREAKPOINT — Line 112: inspect hidden1 vs hidden1_raw
# Compare hidden1 to hidden1_raw:
#   hidden1_raw: [-0.3, 0.5, -1.2, 0.8, ...]
#   hidden1:     [ 0.0, 0.5,  0.0, 0.8, ...]  ← negatives became 0
# Try in Evaluate: (hidden1_raw < 0).sum(), (hidden1 == 0).sum()

# Layer 2: 16 → 8
hidden2_raw = model.layer2(hidden1)
hidden2 = torch.relu(hidden2_raw)
print(f"\nLayer 2 output: shape {hidden2.shape}")
print(f"  Values: {hidden2.data.tolist()}")
x = 4  # 🔴 BREAKPOINT — Line 123: inspect hidden2
# Shape [8] — 8 numbers. Each is a COMBINATION of the 16 values from layer 1.
# Layer 2 finds patterns in layer 1's patterns — higher-level features.

# Layer 3: 8 → 1 (final prediction)
output_raw = model.layer3(hidden2)
output = torch.sigmoid(output_raw)
print(f"\nFinal output: {output.item():.4f} (probability of FAIL)")
x = 5  # 🔴 BREAKPOINT — Line 131: inspect output_raw, output
# output_raw is the raw logit (any number)
# output is after sigmoid (0 to 1)
# With random weights, this is probably ~0.5 (no knowledge yet).

# The data flow:
# [8] → layer1 → [16] → relu → [16] → layer2 → [8] → relu → [8] → layer3 → [1] → sigmoid → prediction
print(f"\nData flow:")
print(f"  Input:    {features.shape}  →  8 features")
print(f"  Layer 1:  {hidden1.shape} →  16 hidden neurons")
print(f"  Layer 2:  {hidden2.shape}  →  8 hidden neurons")
print(f"  Output:   {output.shape}   →  1 prediction")


# ============================================================
# PART 3: Count the parameters — this IS the model
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Model parameters")
print(f"{'='*60}")

total_params = 0
for name, param in model.named_parameters():
    print(f"  {name:25s}  shape: {str(param.shape):15s}  params: {param.numel()}")
    total_params += param.numel()

print(f"\n  Total parameters: {total_params}")
print(f"  (Qwen 14B has 14,000,000,000 — same concept, just bigger)")
x = 6  # 🔴 BREAKPOINT — Line 153: inspect total_params
# layer1.weight: [16, 8]  = 128 parameters
# layer1.bias:   [16]     = 16 parameters
# layer2.weight: [8, 16]  = 128 parameters
# layer2.bias:   [8]      = 8 parameters
# layer3.weight: [1, 8]   = 8 parameters
# layer3.bias:   [1]      = 1 parameter
# Total: 289 parameters
#
# Our classifier: 289 parameters
# Qwen 14B: 14,000,000,000 parameters
# Same idea, different scale. Every parameter is a number in a tensor.


# ============================================================
# PART 4: Train the network
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Training")
print(f"{'='*60}")

# More training data
diffs = [
    ("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'", 1.0),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", 0.0),
    ("+ const SECRET = 'password123'\n+ eval(input)", 1.0),
    ("+ import { DomSanitizer }\n+ constructor(private sanitizer: DomSanitizer)", 0.0),
    ("+ innerHTML = userInput\n+ API_KEY = 'ghp_abc'", 1.0),
    ("+ import { Observable }\n+ subscribe(data =>", 0.0),
]

training_data = [(extract_features(diff), torch.tensor([label])) for diff, label in diffs]

# PyTorch's built-in optimizer and loss — same as Session 3 but cleaner
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # SGD = Stochastic Gradient Descent
loss_fn = nn.BCELoss()  # Binary Cross-Entropy (same as Session 3's manual formula)

# Training loop
for epoch in range(200):
    total_loss = 0.0

    for features, label in training_data:
        # Forward pass
        prediction = model(features)

        # Compute loss
        loss = loss_fn(prediction, label)
        total_loss += loss.item()

        # Backward pass + update (PyTorch does it all)
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute new gradients
        optimizer.step()        # Update weights

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss = {total_loss:.4f}")

x = 7  # 🔴 BREAKPOINT — Line 197: inspect total_loss after training
# Watch loss decrease: epoch 0 ~4.0 → epoch 200 ~0.01
# The network learned to distinguish bad code from good code.
# Try in Evaluate: total_loss


# ============================================================
# PART 5: Test the trained network
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Predictions after training")
print(f"{'='*60}")

test_cases = [
    ("+ eval(userInput)\n+ const SECRET = 'abc'", "Should FAIL"),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", "Should PASS"),
    ("+ innerHTML = data\n+ API_KEY = 'sk-test'", "Should FAIL"),
    ("+ import { DomSanitizer }\n+ import { Observable }", "Should PASS"),
]

print("\nTest predictions:")
for diff, expected in test_cases:
    features = extract_features(diff)
    with torch.no_grad():  # No gradient tracking for inference
        pred = model(features)
    verdict = "FAIL" if pred.item() > 0.5 else "PASS"
    confidence = pred.item() if pred.item() > 0.5 else 1 - pred.item()
    print(f"  {verdict} ({confidence:.1%} confident) — {expected}")
    print(f"    Features: {features.tolist()}")

x = 8  # 🔴 BREAKPOINT — Line 219: inspect predictions
# The network should correctly classify:
#   eval + secrets → FAIL (high confidence)
#   HttpClient + subscribe → PASS (high confidence)
#
# Compare to Session 3's single neuron:
#   Session 3: one weight per feature, straight line boundary
#   Session 4: 289 parameters, can learn COMBINATIONS of features
#
# The network doesn't just know "eval = bad" — it knows
# "eval + innerHTML together is WORSE than eval alone"


print(f"\n{'='*60}")
print("Session 4 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. A layer of N neurons = a [N, inputs] weight matrix + [N] bias
2. nn.Linear(8, 16) creates 16 neurons, each reading 8 inputs
3. ReLU activation (zero negatives) adds non-linearity between layers
4. Without activation, stacking layers = one big matrix multiply (pointless)
5. model(input) calls forward() — data flows through all layers
6. optimizer.zero_grad() → loss.backward() → optimizer.step() = one training step
7. Parameter count = model size (ours: 289, Qwen: 14B, same concept)

This is a real neural network. Everything from here scales up:
  - More layers = deeper
  - More neurons per layer = wider
  - More parameters = more capacity to learn

Next session: Forward Pass deep dive — watching data transform layer by layer
""")
