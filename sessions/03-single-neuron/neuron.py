"""
Session 3: A Single Neuron

A neuron is: dot_product(weights, features) + bias → activation
That's it. Everything in deep learning is built from this.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import re

# ============================================================
# First, let's get our features from Session 2
# ============================================================

def extract_features(diff: str) -> torch.Tensor:
    """Same hand-crafted features from Session 2."""
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


bad_diff = """
+ const API_KEY = 'sk-proj-abc123def456';
+ const parsed = eval('(' + input + ')');
+ document.getElementById('output').innerHTML = parsed.html;
"""

good_diff = """
+ import { HttpClient } from '@angular/common/http';
+ constructor(private http: HttpClient) {}
+ this.http.get('/api/settings').subscribe(data => {});
"""

bad_features = extract_features(bad_diff)
good_features = extract_features(good_diff)

print("Features (bad diff):", bad_features.tolist())
print("Features (good diff):", good_features.tolist())
# bad:  [1, 1, 1, 1, 0, 0, 0, 0]  — has eval, innerHTML, secrets
# good: [0, 0, 0, 0, 1, 1, 1, 1]  — has imports, constructor, subscribe, HttpClient


# ============================================================
# PART 1: Create a single neuron (random weights)
# ============================================================

NUM_FEATURES = 8

# These are the neuron's parameters — random to start
# requires_grad=True means PyTorch will track operations for training
torch.manual_seed(42)  # Fixed seed so results are reproducible
weights = torch.randn(NUM_FEATURES, requires_grad=True)  # 8 random weights
bias = torch.zeros(1, requires_grad=True)                 # bias starts at 0

print(f"\n{'='*60}")
print("PART 1: A neuron with random weights")
print(f"{'='*60}")
print(f"Weights: {weights.data.tolist()}")
print(f"Bias: {bias.data.item():.4f}")
x = 1  # 🔴 BREAKPOINT — Line 56: inspect weights and bias
# weights is a tensor of 8 random numbers — one per feature.
# Each weight says "how important is this feature?"
# Right now they're random — the neuron knows NOTHING.
# Try in Evaluate: weights.shape, weights.requires_grad


# ============================================================
# PART 2: Make a prediction (forward pass)
# ============================================================

# The neuron's computation: dot_product(weights, features) + bias
raw_output = torch.dot(weights, bad_features) + bias

print(f"\nPredicting on BAD diff:")
print(f"  weights · features = {torch.dot(weights, bad_features).item():.4f}")
print(f"  + bias = {raw_output.item():.4f}")
x = 2  # 🔴 BREAKPOINT — Line 72: inspect raw_output
# raw_output could be ANY number: -3.7, 0.2, 15.6, etc.
# But we need a probability between 0 and 1:
#   0.0 = definitely PASS
#   1.0 = definitely FAIL
# That's what the activation function does.

# Sigmoid activation: squishes any number to the range (0, 1)
# Large positive → close to 1.0
# Large negative → close to 0.0
# Zero → exactly 0.5
prediction = torch.sigmoid(raw_output)

print(f"  sigmoid({raw_output.item():.4f}) = {prediction.item():.4f}")
print(f"  Interpretation: {prediction.item():.1%} chance of FAIL")
x = 3  # 🔴 BREAKPOINT — Line 85: inspect prediction
# prediction is now between 0 and 1.
# With random weights, it's probably around 0.5 (coin flip — no knowledge).
# Try in Evaluate: torch.sigmoid(torch.tensor(10.0))   → ~1.0
#                  torch.sigmoid(torch.tensor(-10.0))  → ~0.0
#                  torch.sigmoid(torch.tensor(0.0))    → 0.5


# ============================================================
# PART 3: How wrong is it? (loss)
# ============================================================

# The truth: bad_diff SHOULD be flagged (label = 1.0 = FAIL)
label = torch.tensor(1.0)

# Binary Cross-Entropy loss:
# If label=1 and prediction=0.9 → small loss (correct!)
# If label=1 and prediction=0.1 → large loss (wrong!)
loss = -( label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction) )

print(f"\nLoss calculation:")
print(f"  True label: {label.item():.0f} (FAIL)")
print(f"  Prediction: {prediction.item():.4f}")
print(f"  Loss: {loss.item():.4f}")
print(f"  (Lower loss = better prediction)")
x = 4  # 🔴 BREAKPOINT — Line 105: inspect loss, prediction, label
# If prediction is close to 1.0 (correct for FAIL), loss is small.
# If prediction is far from 1.0, loss is large.
# Try in Evaluate:
#   -torch.log(torch.tensor(0.9))  → 0.105 (small loss, good prediction)
#   -torch.log(torch.tensor(0.1))  → 2.302 (large loss, bad prediction)
#   -torch.log(torch.tensor(0.5))  → 0.693 (medium loss, uncertain)


# ============================================================
# PART 4: Which weights need to change? (backward pass)
# ============================================================

# This is the magic: compute gradients for ALL parameters at once
loss.backward()

print(f"\nGradients (how much to change each weight):")
for i, (w, g, f) in enumerate(zip(weights.data, weights.grad, bad_features)):
    direction = "↑ increase" if g < 0 else "↓ decrease"
    print(f"  weight[{i}]: {w:.4f}  grad: {g:.4f}  feature: {f:.0f}  → {direction}")
print(f"  bias grad: {bias.grad.item():.4f}")
x = 5  # 🔴 BREAKPOINT — Line 122: inspect weights.grad
# The gradient for each weight tells you:
#   - Negative gradient → increase this weight (prediction was too low)
#   - Positive gradient → decrease this weight (prediction was too high)
#   - Large gradient → this weight needs a BIG change
#   - Zero gradient → this feature was 0 (didn't contribute), no change needed
#
# Notice: features that were 0 have 0 gradient — they weren't used,
# so there's nothing to learn about them from this example.


# ============================================================
# PART 5: Update the weights (one learning step)
# ============================================================

LEARNING_RATE = 0.1  # How big a step to take

print(f"\nUpdating weights (learning rate = {LEARNING_RATE}):")
print(f"  Before: weights = {weights.data.tolist()}")

# The update rule: new_weight = old_weight - learning_rate * gradient
with torch.no_grad():  # Don't track this operation
    weights -= LEARNING_RATE * weights.grad
    bias -= LEARNING_RATE * bias.grad

# Clear gradients for next round
weights.grad.zero_()
bias.grad.zero_()

print(f"  After:  weights = {weights.data.tolist()}")

# Now predict again with updated weights
raw_output_2 = torch.dot(weights, bad_features) + bias
prediction_2 = torch.sigmoid(raw_output_2)
print(f"\n  Prediction before update: {prediction.item():.4f}")
print(f"  Prediction after update:  {prediction_2.item():.4f}")
print(f"  (Should be closer to 1.0 = FAIL)")
x = 6  # 🔴 BREAKPOINT — Line 142: inspect weights, prediction_2
# The weights changed! And the prediction moved toward the correct answer.
# This is ONE step of training. Real training does this thousands of times.
# Try in Evaluate: prediction_2.item() — it should be closer to 1.0 than before


# ============================================================
# PART 6: Full training loop — many steps
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Training loop (100 steps)")
print(f"{'='*60}")

# Reset weights
torch.manual_seed(42)
weights = torch.randn(NUM_FEATURES, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

# Training data: two examples
training_data = [
    (bad_features, torch.tensor(1.0)),    # bad diff → FAIL (1)
    (good_features, torch.tensor(0.0)),   # good diff → PASS (0)
]

LEARNING_RATE = 0.5

for step in range(100):
    total_loss = 0.0

    for features, label in training_data:
        # Forward pass
        raw = torch.dot(weights, features) + bias
        pred = torch.sigmoid(raw)

        # Loss
        loss = -(label * torch.log(pred + 1e-8) + (1 - label) * torch.log(1 - pred + 1e-8))
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        with torch.no_grad():
            weights -= LEARNING_RATE * weights.grad
            bias -= LEARNING_RATE * bias.grad
        weights.grad.zero_()
        bias.grad.zero_()

    if step % 10 == 0:
        # Test predictions
        with torch.no_grad():
            bad_pred = torch.sigmoid(torch.dot(weights, bad_features) + bias)
            good_pred = torch.sigmoid(torch.dot(weights, good_features) + bias)
        print(f"  Step {step:3d}: loss={total_loss:.4f}  bad_pred={bad_pred.item():.4f}(want 1.0)  good_pred={good_pred.item():.4f}(want 0.0)")

x = 7  # 🔴 BREAKPOINT — Line 195: inspect weights after training
# Watch the predictions converge:
#   Step 0:   bad_pred ≈ 0.5 (random), good_pred ≈ 0.5 (random)
#   Step 50:  bad_pred → 1.0 (correct!), good_pred → 0.0 (correct!)
#   Step 100: both very confident
#
# The neuron LEARNED to distinguish bad code from good code.
# Try in Evaluate: weights.data.tolist() — see which weights are large


# ============================================================
# PART 7: What did the neuron learn?
# ============================================================

print(f"\n{'='*60}")
print("PART 7: What the neuron learned")
print(f"{'='*60}")

feature_names = [
    "eval_calls", "innerHTML", "secret_names", "secret_patterns",
    "imports", "constructor", "reactive", "angular_services"
]

print("\nLearned weights:")
for name, w in zip(feature_names, weights.data):
    bar = "█" * int(abs(w.item()) * 3)
    direction = "FAIL ↑" if w > 0 else "PASS ↓"
    print(f"  {name:20s}: {w.item():+.4f}  {bar}  ({direction})")

print(f"\n  Bias: {bias.data.item():+.4f}")

x = 8  # 🔴 BREAKPOINT — Line 219: inspect weights with feature names
# Positive weights → this feature pushes toward FAIL
# Negative weights → this feature pushes toward PASS
#
# You should see:
#   eval_calls:     LARGE POSITIVE (eval = bad)
#   innerHTML:      LARGE POSITIVE (innerHTML = bad)
#   secret_names:   LARGE POSITIVE (secrets = bad)
#   angular_services: LARGE NEGATIVE (proper Angular = good)
#
# The neuron discovered what our code review standards say!
# It learned: "eval + secrets = FAIL, HttpClient + subscribe = PASS"

# Final predictions
with torch.no_grad():
    bad_pred = torch.sigmoid(torch.dot(weights, bad_features) + bias)
    good_pred = torch.sigmoid(torch.dot(weights, good_features) + bias)

print(f"\nFinal predictions:")
print(f"  Bad diff:  {bad_pred.item():.4f} → {'FAIL' if bad_pred > 0.5 else 'PASS'}")
print(f"  Good diff: {good_pred.item():.4f} → {'FAIL' if good_pred > 0.5 else 'PASS'}")


print(f"\n{'='*60}")
print("Session 3 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. A neuron computes: dot(weights, features) + bias
2. Sigmoid squishes the output to a 0-1 probability
3. Loss measures how wrong the prediction is
4. Gradients tell us which weights to change and by how much
5. Training = repeat (predict → measure loss → compute gradients → update weights)
6. After training, the weights ENCODE knowledge (eval=bad, HttpClient=good)

This single neuron IS a model. A neural network is just many of these
stacked together — that's Session 4.

Next session: A Neural Network — stacking neurons into layers
""")
