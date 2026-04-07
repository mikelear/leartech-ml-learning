"""
Session 7: Backpropagation Through Multiple Layers

Session 6 showed the chain rule for ONE neuron.
This session: gradients flowing backwards through a FULL network.

The key question: how does the loss signal reach the earliest layers?
Answer: the chain rule multiplied through each layer in reverse.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import re


def extract_features(diff):
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
# PART 1: The computation graph
# ============================================================

class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        self.h1_raw = self.layer1(x)
        self.h1 = torch.relu(self.h1_raw)
        self.h2_raw = self.layer2(self.h1)
        self.h2 = torch.relu(self.h2_raw)
        self.out_raw = self.layer3(self.h2)
        self.out = torch.sigmoid(self.out_raw)
        return self.out


torch.manual_seed(42)
model = ThreeLayerNet()
loss_fn = nn.BCELoss()

features = extract_features("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'")
label = torch.tensor([1.0])

# Forward pass — PyTorch records EVERY operation
prediction = model(features)
loss = loss_fn(prediction, label)

print("=" * 60)
print("PART 1: The computation graph")
print("=" * 60)
print(f"\nPrediction: {prediction.item():.4f}")
print(f"Loss: {loss.item():.4f}")
print(f"\nComputation graph (what backward will walk):")
print(f"  loss.grad_fn: {loss.grad_fn}")
print(f"  → {loss.grad_fn.next_functions}")
x = 1  # 🔴 BREAKPOINT — Line 62: inspect loss.grad_fn
# Expand loss.grad_fn in Variables pane.
# It shows the chain: BinaryCrossEntropyBackward → SigmoidBackward → ...
# Each node knows how to compute its local gradient.
# backward() walks this chain in reverse.


# ============================================================
# PART 2: Backward pass — gradients for ALL layers
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Gradients after backward()")
print(f"{'='*60}")

loss.backward()

# Every weight in every layer now has a gradient
print("\nGradients per layer:")
for name, param in model.named_parameters():
    grad = param.grad
    print(f"  {name:20s}  shape: {str(param.shape):12s}  "
          f"grad_mean: {grad.mean().item():+.6f}  "
          f"grad_max: {grad.abs().max().item():.6f}")

x = 2  # 🔴 BREAKPOINT — Line 80: inspect model.layer1.weight.grad, model.layer3.weight.grad
# Compare gradient SIZES across layers:
#   layer3.weight.grad — LARGEST (closest to loss, strongest signal)
#   layer1.weight.grad — SMALLEST (furthest from loss, weakest signal)
#
# Try in Evaluate:
#   model.layer3.weight.grad.abs().mean()  vs  model.layer1.weight.grad.abs().mean()
# Layer 3 gradients are bigger — it learns faster.


# ============================================================
# PART 3: Gradient flow visualisation
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Gradient magnitude through layers")
print(f"{'='*60}")

layer_names = []
grad_magnitudes = []

for name, param in model.named_parameters():
    if 'weight' in name:
        mag = param.grad.abs().mean().item()
        layer_names.append(name)
        grad_magnitudes.append(mag)
        bar = "█" * int(mag * 500)
        print(f"  {name:20s}: {mag:.6f}  {bar}")

# Ratio: how much smaller are layer1 gradients vs layer3?
if grad_magnitudes[0] > 0 and grad_magnitudes[-1] > 0:
    ratio = grad_magnitudes[-1] / grad_magnitudes[0]
    print(f"\n  Layer3/Layer1 gradient ratio: {ratio:.1f}x")
    print(f"  Layer 3 gets {ratio:.1f}× stronger learning signal than Layer 1")

x = 3  # 🔴 BREAKPOINT — Line 104: inspect grad_magnitudes
# The gradient shrinks as it flows backwards through layers.
# Layer 3: strong signal → learns fast
# Layer 1: weak signal → learns slow
# This is the VANISHING GRADIENT problem.


# ============================================================
# PART 4: Vanishing gradients — sigmoid vs ReLU
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Why ReLU helps — sigmoid vs ReLU gradients")
print(f"{'='*60}")

# Sigmoid derivative: sig(x) * (1 - sig(x))
# Maximum value: 0.25 (at x=0). Always < 1.
# Each layer MULTIPLIES by this → gradients shrink exponentially!

# ReLU derivative: 0 (if x<0) or 1 (if x>0)
# Value is either 0 or 1. Never shrinks the gradient!

print("\nSigmoid derivative at different inputs:")
for val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
    t = torch.tensor(val)
    sig = torch.sigmoid(t)
    deriv = sig * (1 - sig)
    print(f"  sigmoid'({val:+.1f}) = {deriv.item():.4f}")

print("\nReLU derivative at different inputs:")
for val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
    deriv = 1.0 if val > 0 else 0.0
    print(f"  relu'({val:+.1f}) = {deriv:.4f}")

# Demo: deep network with sigmoid vs ReLU
class DeepSigmoidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.Sigmoid(),
            nn.Linear(16, 16), nn.Sigmoid(),
            nn.Linear(16, 16), nn.Sigmoid(),
            nn.Linear(16, 16), nn.Sigmoid(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.layers(x)

class DeepReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),  # Only final layer uses sigmoid
        )
    def forward(self, x):
        return self.layers(x)

torch.manual_seed(42)
sig_net = DeepSigmoidNet()
relu_net = DeepReLUNet()

# Forward + backward on both
sig_pred = sig_net(features)
sig_loss = loss_fn(sig_pred, label)
sig_loss.backward()

relu_pred = relu_net(features)
relu_loss = loss_fn(relu_pred, label)
relu_loss.backward()

print(f"\n5-layer network — gradient magnitude at FIRST layer:")
sig_grad = list(sig_net.parameters())[0].grad.abs().mean().item()
relu_grad = list(relu_net.parameters())[0].grad.abs().mean().item()
print(f"  Sigmoid network: {sig_grad:.8f}")
print(f"  ReLU network:    {relu_grad:.8f}")
if sig_grad > 0:
    print(f"  ReLU has {relu_grad/sig_grad:.0f}× stronger gradient at layer 1")

x = 4  # 🔴 BREAKPOINT — Line 161: inspect sig_grad, relu_grad
# ReLU's gradient at layer 1 is MUCH bigger than sigmoid's.
# With sigmoid, each layer multiplies by ≤0.25 → after 5 layers: 0.25^5 ≈ 0.001
# With ReLU, each layer multiplies by 0 or 1 → gradient passes through unchanged
# This is why modern networks use ReLU (or variants) in hidden layers.


# ============================================================
# PART 5: Manual backprop — layer by layer
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Manual backpropagation through 3 layers")
print(f"{'='*60}")

# Reset the model and do a fresh forward pass
torch.manual_seed(42)
model = ThreeLayerNet()
prediction = model(features)
loss = loss_fn(prediction, label)

# We'll compute gradients manually, then verify against PyTorch

# --- Backward through loss ---
# d(BCELoss)/d(prediction) = -(truth/pred - (1-truth)/(1-pred))
dloss_dpred = -(label / prediction - (1 - label) / (1 - prediction))
print(f"\nd(loss)/d(prediction) = {dloss_dpred.item():.6f}")

# --- Backward through sigmoid (layer3 output) ---
# d(sigmoid)/d(input) = sigmoid * (1 - sigmoid)
dsigmoid = prediction * (1 - prediction)
dloss_dout_raw = dloss_dpred * dsigmoid
print(f"d(loss)/d(out_raw) = {dloss_dout_raw.item():.6f}")

# --- Backward through layer3: out_raw = W3 @ h2 + b3 ---
# d(out_raw)/d(W3) = h2
# d(out_raw)/d(h2) = W3
grad_W3_manual = dloss_dout_raw.unsqueeze(0) * model.h2.unsqueeze(0)  # outer product
grad_b3_manual = dloss_dout_raw
dloss_dh2 = (model.layer3.weight.T @ dloss_dout_raw.unsqueeze(1)).squeeze()
print(f"d(loss)/d(h2) shape: {dloss_dh2.shape}  mean: {dloss_dh2.mean().item():.6f}")

# --- Backward through ReLU (between layer2 and layer3) ---
# d(relu)/d(input) = 1 if input > 0, else 0
relu2_mask = (model.h2_raw > 0).float()
dloss_dh2_raw = dloss_dh2 * relu2_mask
print(f"ReLU mask (layer2): {relu2_mask.tolist()}")
print(f"d(loss)/d(h2_raw): {dloss_dh2_raw.mean().item():.6f}  (some zeroed by ReLU)")

# --- Backward through layer2: h2_raw = W2 @ h1 + b2 ---
grad_W2_manual = dloss_dh2_raw.unsqueeze(1) @ model.h1.unsqueeze(0)
dloss_dh1 = (model.layer2.weight.T @ dloss_dh2_raw.unsqueeze(1)).squeeze()
print(f"d(loss)/d(h1) shape: {dloss_dh1.shape}  mean: {dloss_dh1.mean().item():.6f}")

# --- Backward through ReLU (between layer1 and layer2) ---
relu1_mask = (model.h1_raw > 0).float()
dloss_dh1_raw = dloss_dh1 * relu1_mask

# --- Backward through layer1: h1_raw = W1 @ input + b1 ---
grad_W1_manual = dloss_dh1_raw.unsqueeze(1) @ features.unsqueeze(0)

x = 5  # 🔴 BREAKPOINT — Line 196: inspect all the manual gradients
# You just computed backprop by hand!
# dloss_dpred → dsigmoid → dloss_dout_raw → layer3 → relu → layer2 → relu → layer1
# Each step multiplied by the local derivative (chain rule).
#
# Try in Evaluate: grad_W1_manual.shape, grad_W3_manual.shape


# ============================================================
# PART 6: Verify manual == PyTorch
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Verify manual backprop matches PyTorch")
print(f"{'='*60}")

# Now let PyTorch do it
loss.backward()

# Compare
match_W3 = torch.allclose(grad_W3_manual, model.layer3.weight.grad, atol=1e-5)
match_W2 = torch.allclose(grad_W2_manual, model.layer2.weight.grad, atol=1e-5)
match_W1 = torch.allclose(grad_W1_manual, model.layer1.weight.grad, atol=1e-5)

print(f"\n  Layer 3 weights grad match: {match_W3}")
print(f"  Layer 2 weights grad match: {match_W2}")
print(f"  Layer 1 weights grad match: {match_W1}")

if match_W3 and match_W2 and match_W1:
    print("\n  ✓ All gradients match! Our manual backprop is correct.")
else:
    print("\n  ✗ Mismatch — check the manual computation")
    print(f"    W3 manual max: {grad_W3_manual.abs().max():.6f}  pytorch: {model.layer3.weight.grad.abs().max():.6f}")
    print(f"    W1 manual max: {grad_W1_manual.abs().max():.6f}  pytorch: {model.layer1.weight.grad.abs().max():.6f}")

x = 6  # 🔴 BREAKPOINT — Line 226: inspect the match results
# If all True: you just proved you understand what backward() does.
# Your manual chain rule computation produced the SAME gradients
# as PyTorch's autograd engine.


# ============================================================
# PART 7: Watch gradients during training
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Gradient magnitude during training")
print(f"{'='*60}")

torch.manual_seed(42)
model = ThreeLayerNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

diffs = [
    ("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'", 1.0),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", 0.0),
    ("+ const SECRET = 'password123'\n+ eval(input)", 1.0),
    ("+ import { DomSanitizer }\n+ constructor(private sanitizer: DomSanitizer)", 0.0),
]
training_data = [(extract_features(d), torch.tensor([l])) for d, l in diffs]

print(f"\n{'Epoch':>6s}  {'Loss':>8s}  {'L1 grad':>10s}  {'L2 grad':>10s}  {'L3 grad':>10s}")
print(f"{'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

for epoch in range(50):
    total_loss = 0
    for feat, lab in training_data:
        optimizer.zero_grad()
        pred = model(feat)
        loss = loss_fn(pred, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        g1 = model.layer1.weight.grad.abs().mean().item()
        g2 = model.layer2.weight.grad.abs().mean().item()
        g3 = model.layer3.weight.grad.abs().mean().item()
        print(f"  {epoch:>4d}  {total_loss:>8.4f}  {g1:>10.6f}  {g2:>10.6f}  {g3:>10.6f}")

x = 7  # 🔴 BREAKPOINT — Line 267: inspect gradient magnitudes over training
# Watch:
#   - Loss decreases (model is learning)
#   - Layer 3 gradients are largest (learns fastest)
#   - Layer 1 gradients are smallest (learns slowest)
#   - As loss → 0, ALL gradients → 0 (nothing more to learn)


print(f"\n{'='*60}")
print("Session 7 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. Backprop = chain rule applied layer by layer in REVERSE
2. Each layer: receive gradient from ahead, multiply by local derivative, pass backwards
3. Gradients SHRINK as they flow backwards (vanishing gradient problem)
4. ReLU helps: derivative is 0 or 1 (never shrinks). Sigmoid: max derivative 0.25 (always shrinks)
5. Layer closest to loss learns fastest, earliest layer learns slowest
6. Manual backprop = PyTorch's backward() — same maths, just automated
7. As training converges (loss → 0), gradients → 0 (nothing left to learn)

You now understand the COMPLETE training cycle:
  Session 2: text → features (input)
  Session 5: features → prediction (forward pass)
  Session 6: prediction vs truth → loss (how wrong)
  Session 7: loss → gradients for every weight (backward pass)
  Session 8: gradients → weight updates (optimizer)

Next session: Training Loop — putting it all together with batches, epochs, and learning rate
""")
