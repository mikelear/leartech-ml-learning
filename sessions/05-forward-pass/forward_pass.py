"""
Session 5: Forward Pass Deep Dive

Zooming into what ACTUALLY happens when data flows through a network.
We'll trace specific numbers through each layer and see why the
network makes the prediction it does.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import re


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


class CodeReviewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x


# ============================================================
# PART 1: Trace one input through the network
# ============================================================

torch.manual_seed(42)
model = CodeReviewNet()

bad_diff = "+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'"
features = extract_features(bad_diff)

feature_names = [
    "eval", "innerHTML", "secrets", "secret_patterns",
    "imports", "constructor", "reactive", "angular_services"
]

print("=" * 60)
print("PART 1: Tracing input through the network")
print("=" * 60)
print(f"\nInput features:")
for name, val in zip(feature_names, features):
    print(f"  {name:20s}: {val.item():.0f}")
x = 1  # 🔴 BREAKPOINT — Line 58: inspect features
# features = [1, 1, 1, 1, 0, 0, 0, 0]
# The first 4 are "bad" signals, the last 4 are "good" signals (all zero)


# ============================================================
# PART 2: Inside one neuron (neuron 0 of layer 1)
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Inside one neuron (layer1, neuron 0)")
print(f"{'='*60}")

# Get neuron 0's weights — one row of the weight matrix
neuron0_weights = model.layer1.weight[0]  # Shape [8] — one weight per input feature
neuron0_bias = model.layer1.bias[0]       # Shape [] — one bias

print(f"\nNeuron 0 weights:")
for name, w, f in zip(feature_names, neuron0_weights, features):
    contribution = w.item() * f.item()
    print(f"  {name:20s}: weight={w.item():+.4f} × feature={f.item():.0f} = {contribution:+.4f}")

dot_product = torch.dot(neuron0_weights, features)
raw_output = dot_product + neuron0_bias
after_relu = torch.relu(raw_output)

print(f"\n  Dot product: {dot_product.item():.4f}")
print(f"  + bias {neuron0_bias.item():.4f} = {raw_output.item():.4f}")
print(f"  After ReLU: {after_relu.item():.4f}")
x = 2  # 🔴 BREAKPOINT — Line 87: inspect neuron0_weights, dot_product, raw_output, after_relu
# This is EXACTLY what Session 3's single neuron did.
# Each weight multiplies one feature. Sum them up. Add bias. Apply ReLU.
#
# Try in Evaluate:
#   torch.dot(neuron0_weights, features)  — the weighted sum
#   neuron0_weights * features            — element-wise products (before summing)


# ============================================================
# PART 3: All 16 neurons in layer 1
# ============================================================

print(f"\n{'='*60}")
print("PART 3: All 16 neurons in layer 1")
print(f"{'='*60}")

# This is what model.layer1(features) does internally:
# matrix multiply: [16, 8] @ [8] = [16], then add [16] bias
layer1_raw = model.layer1.weight @ features + model.layer1.bias
layer1_activated = torch.relu(layer1_raw)

print(f"\nLayer 1 output (16 neurons):")
print(f"  {'Neuron':>8s}  {'Raw':>8s}  {'After ReLU':>10s}  {'Status':>8s}")
print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}")
for i, (raw, activated) in enumerate(zip(layer1_raw, layer1_activated)):
    status = "FIRES" if activated > 0 else "DEAD"
    bar = "█" * int(activated.item() * 5) if activated > 0 else ""
    print(f"  {i:>8d}  {raw.item():>8.4f}  {activated.item():>10.4f}  {status:>8s}  {bar}")

alive = (layer1_activated > 0).sum().item()
dead = (layer1_activated == 0).sum().item()
print(f"\n  {alive:.0f} neurons fire, {dead:.0f} are dead (zeroed by ReLU)")
x = 3  # 🔴 BREAKPOINT — Line 109: inspect layer1_raw, layer1_activated
# Some neurons fire (positive after ReLU), some are dead (negative → zeroed).
# The PATTERN of which neurons fire is different for different inputs.
# Bad code activates certain neurons. Good code activates different ones.
# This is how the network encodes information.
#
# Try in Evaluate:
#   (layer1_activated > 0).sum()  — how many fire
#   layer1_activated.max()        — strongest activation


# ============================================================
# PART 4: Layer 2 — combining layer 1's outputs
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Layer 2 — combining patterns")
print(f"{'='*60}")

# Layer 2 reads ALL 16 outputs from layer 1
# Each of its 8 neurons computes: dot(weights[8,16], layer1_output[16]) + bias
layer2_raw = model.layer2.weight @ layer1_activated + model.layer2.bias
layer2_activated = torch.relu(layer2_raw)

print(f"\nLayer 2 output (8 neurons):")
for i, (raw, activated) in enumerate(zip(layer2_raw, layer2_activated)):
    status = "FIRES" if activated > 0 else "DEAD"
    print(f"  Neuron {i}: raw={raw.item():+.4f}  activated={activated.item():.4f}  {status}")
x = 4  # 🔴 BREAKPOINT — Line 126: inspect layer2_activated
# Layer 2's neurons each look at ALL 16 neurons from layer 1.
# They're finding COMBINATIONS — "neuron 3 AND neuron 7 both fired"
# is a higher-level pattern than any single layer 1 neuron.


# ============================================================
# PART 5: The final neuron — making the prediction
# ============================================================

print(f"\n{'='*60}")
print("PART 5: The final prediction")
print(f"{'='*60}")

# One neuron reads all 8 outputs from layer 2
final_raw = model.layer3.weight @ layer2_activated + model.layer3.bias
prediction = torch.sigmoid(final_raw)

print(f"\nFinal neuron:")
print(f"  Reads layer 2: {layer2_activated.data.tolist()}")
print(f"  Weights:       {model.layer3.weight.data[0].tolist()}")
print(f"  Raw output:    {final_raw.item():.4f}")
print(f"  After sigmoid: {prediction.item():.4f}")
print(f"  Verdict:       {'FAIL' if prediction.item() > 0.5 else 'PASS'} ({prediction.item():.1%})")
x = 5  # 🔴 BREAKPOINT — Line 143: inspect final_raw, prediction
# The entire forward pass:
#   8 features → 16 neurons → 8 neurons → 1 prediction
#   [8] → [16] → [8] → [1]
# Each arrow is a matrix multiply + activation.
# With random weights, the prediction is meaningless.
# After training, each layer extracts meaningful patterns.

# Verify: model(features) gives the same answer
model_output = model(features)
print(f"\n  model(features) = {model_output.item():.4f}")
print(f"  Manual forward  = {prediction.item():.4f}")
print(f"  Match: {torch.allclose(model_output, prediction)}")


# ============================================================
# PART 6: Train, then trace again — see what changed
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Forward pass AFTER training")
print(f"{'='*60}")

# Quick training (same as Session 4)
diffs = [
    ("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'", 1.0),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", 0.0),
    ("+ const SECRET = 'password123'\n+ eval(input)", 1.0),
    ("+ import { DomSanitizer }\n+ constructor(private sanitizer: DomSanitizer)", 0.0),
    ("+ innerHTML = userInput\n+ API_KEY = 'ghp_abc'", 1.0),
    ("+ import { Observable }\n+ subscribe(data =>", 0.0),
]
training_data = [(extract_features(d), torch.tensor([l])) for d, l in diffs]
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

for epoch in range(300):
    for feat, label in training_data:
        optimizer.zero_grad()
        loss_fn(model(feat), label).backward()
        optimizer.step()

# Now trace the TRAINED network with the same input
layer1_trained = torch.relu(model.layer1(features))
layer2_trained = torch.relu(model.layer2(layer1_trained))
output_trained = torch.sigmoid(model.layer3(layer2_trained))

print(f"\nTrained network — bad diff prediction: {output_trained.item():.4f}")
x = 6  # 🔴 BREAKPOINT — Line 183: inspect layer1_trained, layer2_trained, output_trained
# Compare to the random network (Part 3):
#   - Different neurons fire now (the network reorganised)
#   - The prediction is confident (close to 1.0 for bad code)
# Try in Evaluate: output_trained.item()


# ============================================================
# PART 7: Activation patterns — bad vs good code
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Which neurons fire for bad vs good code?")
print(f"{'='*60}")

good_features = extract_features("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(")

bad_layer1 = torch.relu(model.layer1(features))
good_layer1 = torch.relu(model.layer1(good_features))

print(f"\nLayer 1 activation pattern:")
print(f"  {'Neuron':>8s}  {'Bad code':>10s}  {'Good code':>10s}  {'Difference':>10s}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
for i in range(16):
    bad_val = bad_layer1[i].item()
    good_val = good_layer1[i].item()
    diff = bad_val - good_val
    marker = " ←← BAD" if diff > 0.5 else (" ←← GOOD" if diff < -0.5 else "")
    print(f"  {i:>8d}  {bad_val:>10.4f}  {good_val:>10.4f}  {diff:>+10.4f}{marker}")

x = 7  # 🔴 BREAKPOINT — Line 207: inspect bad_layer1, good_layer1
# Different neurons fire for bad vs good code!
# Some neurons are "bad code detectors" (fire strongly for bad, zero for good)
# Some are "good code detectors" (fire for good, zero for bad)
# The network learned to use different neurons for different patterns.
#
# This is REPRESENTATION LEARNING — the network discovered its own features.
# You didn't tell it which neurons should detect what. It figured it out.


# ============================================================
# PART 8: What if we disable a neuron?
# ============================================================

print(f"\n{'='*60}")
print("PART 8: Disabling neurons — what breaks?")
print(f"{'='*60}")

# Find the neuron that fires most for bad code
most_active_bad = bad_layer1.argmax().item()
print(f"\nMost active neuron for bad code: neuron {most_active_bad}")
print(f"  Activation: {bad_layer1[most_active_bad].item():.4f}")

# Kill it by zeroing its weights
with torch.no_grad():
    original_weights = model.layer1.weight[most_active_bad].clone()
    model.layer1.weight[most_active_bad] = 0
    model.layer1.bias[most_active_bad] = 0

# Predict again
damaged_pred = model(features)
print(f"\n  Original prediction (bad code): {output_trained.item():.4f}")
print(f"  After killing neuron {most_active_bad}:  {damaged_pred.item():.4f}")
print(f"  Change: {abs(output_trained.item() - damaged_pred.item()):.4f}")

# Restore
with torch.no_grad():
    model.layer1.weight[most_active_bad] = original_weights

x = 8  # 🔴 BREAKPOINT — Line 235: inspect damaged_pred vs output_trained
# Killing one neuron changed the prediction.
# Some neurons are MORE important than others.
# This is why model interpretability matters — understanding which
# neurons are responsible for which decisions.
#
# In a 14B parameter model, this same principle applies:
# specific neurons fire for specific concepts.


print(f"\n{'='*60}")
print("Session 5 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. Forward pass = data flowing through layers: input → hidden → output
2. Each layer: matrix multiply (W @ x) + bias + activation (ReLU)
3. Different neurons fire for different inputs — the PATTERN is the representation
4. After training, neurons specialise: some detect "bad code", others "good code"
5. The network discovers its own internal features (representation learning)
6. Killing individual neurons changes predictions — some matter more than others

What you've now seen end-to-end:
  Session 1: tensors (the containers)
  Session 2: features (text → numbers)
  Session 3: one neuron (weight × feature + bias)
  Session 4: network (layers of neurons)
  Session 5: forward pass (data flowing through, neurons specialising)

Next session: Loss Function — measuring HOW WRONG the prediction is
""")
