"""
Session 6: Loss Function — How Wrong Is the Prediction?

Loss is a single number: lower = better prediction.
Training exists to make this number smaller.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import re


# ============================================================
# PART 1: What does loss look like?
# ============================================================

print("=" * 60)
print("PART 1: Loss for different predictions")
print("=" * 60)

truth = 1.0  # The correct answer: FAIL

# Binary Cross-Entropy loss: -( truth * log(pred) + (1-truth) * log(1-pred) )
# When truth=1: loss = -log(pred)
# When truth=0: loss = -log(1-pred)

predictions = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
losses = []

print(f"\nTruth = {truth:.0f} (FAIL)")
print(f"{'Prediction':>12s}  {'Loss':>8s}  {'Meaning':>20s}")
print(f"{'─'*12}  {'─'*8}  {'─'*20}")

for pred in predictions:
    pred_t = torch.tensor(pred)
    loss = -torch.log(pred_t)  # Simplified: truth=1, so loss = -log(pred)
    losses.append(loss.item())
    if pred < 0.3:
        meaning = "Very wrong!"
    elif pred < 0.7:
        meaning = "Uncertain"
    else:
        meaning = "Good prediction"
    bar = "█" * int(loss.item() * 3)
    print(f"  {pred:>10.2f}  {loss.item():>8.4f}  {meaning:>20s}  {bar}")

x = 1  # 🔴 BREAKPOINT — Line 44: inspect predictions and losses
# Key insight: loss is NOT linear.
#   pred=0.9 → loss=0.105  (small — close to correct)
#   pred=0.5 → loss=0.693  (medium — uncertain)
#   pred=0.1 → loss=2.302  (huge — very wrong)
#   pred=0.01 → loss=4.605 (massive — completely wrong)
#
# The log function penalises confident wrong predictions MUCH more.
# Try in Evaluate: -torch.log(torch.tensor(0.5)), -torch.log(torch.tensor(0.01))


# ============================================================
# PART 2: The loss landscape
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Loss landscape — plotting loss vs prediction")
print(f"{'='*60}")

# For truth=1 (FAIL), loss at every possible prediction
pred_range = torch.linspace(0.01, 0.99, 50)
loss_values = -torch.log(pred_range)  # loss when truth=1

# The gradient at each point: which direction is downhill?
# d/d(pred) of -log(pred) = -1/pred
gradient_at_each = -1.0 / pred_range

print(f"\nLoss landscape (truth=1.0):")
print(f"  pred=0.01 → loss={-torch.log(torch.tensor(0.01)).item():.2f} (top of hill)")
print(f"  pred=0.50 → loss={-torch.log(torch.tensor(0.50)).item():.2f} (halfway)")
print(f"  pred=0.99 → loss={-torch.log(torch.tensor(0.99)).item():.2f} (bottom of valley)")
print(f"\n  Training walks downhill: pred moves toward 1.0")

x = 2  # 🔴 BREAKPOINT — Line 66: inspect pred_range, loss_values, gradient_at_each
# loss_values is a curve — steep on the left (pred near 0), flat on the right (pred near 1)
# gradient_at_each tells you the slope at each point — steep = big gradient = fast learning
# Try in Evaluate: loss_values[0], loss_values[-1], gradient_at_each[0], gradient_at_each[-1]


# ============================================================
# PART 3: Gradients — which way is downhill?
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Gradient direction")
print(f"{'='*60}")

# Let's compute gradients the PyTorch way
pred_tracked = torch.tensor(0.3, requires_grad=True)
truth_t = torch.tensor(1.0)

# Loss
loss = -(truth_t * torch.log(pred_tracked) + (1 - truth_t) * torch.log(1 - pred_tracked))

print(f"\n  Prediction: {pred_tracked.item():.4f}")
print(f"  Truth: {truth_t.item():.0f}")
print(f"  Loss: {loss.item():.4f}")

# Compute gradient
loss.backward()

print(f"  Gradient: {pred_tracked.grad.item():.4f}")
print(f"  Direction: {'increase prediction' if pred_tracked.grad < 0 else 'decrease prediction'}")
print(f"  (negative gradient = increase the value to reduce loss)")

x = 3  # 🔴 BREAKPOINT — Line 89: inspect pred_tracked, loss, pred_tracked.grad
# The gradient is NEGATIVE — meaning "increase the prediction" to reduce loss.
# That makes sense: truth is 1.0, prediction is 0.3, so we need to go UP.
#
# gradient = -1/pred = -1/0.3 = -3.33
# The further from correct, the larger the gradient (faster learning).
#
# Try different starting predictions in Evaluate:
#   If we started at 0.9, gradient would be -1/0.9 = -1.11 (smaller — less wrong)


# ============================================================
# PART 4: MSE vs BCE — two different loss functions
# ============================================================

print(f"\n{'='*60}")
print("PART 4: MSE vs BCE loss functions")
print(f"{'='*60}")

# Mean Squared Error: (prediction - truth)²
# Binary Cross-Entropy: -(truth * log(pred) + (1-truth) * log(1-pred))

test_preds = torch.linspace(0.01, 0.99, 10)
truth_val = 1.0

print(f"\n  {'Pred':>6s}  {'MSE Loss':>10s}  {'BCE Loss':>10s}  {'MSE grad':>10s}  {'BCE grad':>10s}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

for pred_val in test_preds:
    # MSE
    p_mse = torch.tensor(pred_val.item(), requires_grad=True)
    mse_loss = (p_mse - truth_val) ** 2
    mse_loss.backward()
    mse_grad = p_mse.grad.item()

    # BCE
    p_bce = torch.tensor(pred_val.item(), requires_grad=True)
    bce_loss = -(truth_val * torch.log(p_bce) + (1 - truth_val) * torch.log(1 - p_bce))
    bce_loss.backward()
    bce_grad = p_bce.grad.item()

    print(f"  {pred_val.item():>6.2f}  {mse_loss.item():>10.4f}  {bce_loss.item():>10.4f}  {mse_grad:>10.4f}  {bce_grad:>10.4f}")

x = 4  # 🔴 BREAKPOINT — Line 115: compare MSE and BCE
# Key difference:
#   MSE gradient is SMALL when prediction is very wrong (pred=0.01, grad=-2.0)
#   BCE gradient is HUGE when prediction is very wrong (pred=0.01, grad=-100.0)
#
# BCE "punishes" confident wrong predictions much harder.
# This is why BCE is standard for classification (PASS/FAIL).
# MSE is better for regression (predicting a continuous number).


# ============================================================
# PART 5: Loss on our actual network
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Loss on the network from Sessions 4-5")
print(f"{'='*60}")

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

class CodeReviewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
    def forward(self, x):
        return torch.sigmoid(self.layer3(torch.relu(self.layer2(torch.relu(self.layer1(x))))))

torch.manual_seed(42)
model = CodeReviewNet()
loss_fn = nn.BCELoss()

bad_features = extract_features("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'")
good_features = extract_features("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(")

bad_pred = model(bad_features)
good_pred = model(good_features)

bad_loss = loss_fn(bad_pred, torch.tensor([1.0]))
good_loss = loss_fn(good_pred, torch.tensor([0.0]))
total_loss = bad_loss + good_loss

print(f"\nBefore training:")
print(f"  Bad code:  pred={bad_pred.item():.4f}  truth=1.0  loss={bad_loss.item():.4f}")
print(f"  Good code: pred={good_pred.item():.4f}  truth=0.0  loss={good_loss.item():.4f}")
print(f"  Total loss: {total_loss.item():.4f}")
x = 5  # 🔴 BREAKPOINT — Line 147: inspect bad_pred, good_pred, bad_loss, good_loss, total_loss
# Both predictions are ~0.5 (random). Both losses are ~0.7.
# Total loss is the sum — this is what training minimises.


# ============================================================
# PART 6: Loss across a batch — total error signal
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Training loss over time")
print(f"{'='*60}")

diffs = [
    ("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'", 1.0),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", 0.0),
    ("+ const SECRET = 'password123'\n+ eval(input)", 1.0),
    ("+ import { DomSanitizer }\n+ constructor(private sanitizer: DomSanitizer)", 0.0),
]
training_data = [(extract_features(d), torch.tensor([l])) for d, l in diffs]

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_history = []

for epoch in range(100):
    epoch_loss = 0.0
    for feat, label in training_data:
        optimizer.zero_grad()
        pred = model(feat)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    loss_history.append(epoch_loss)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}: total_loss = {epoch_loss:.4f}")

x = 6  # 🔴 BREAKPOINT — Line 172: inspect loss_history
# loss_history shows the loss DECREASING over epochs.
# Epoch 0: ~2.8 (random predictions, high error)
# Epoch 50: ~0.1 (confident correct predictions, low error)
# Epoch 100: ~0.01 (very confident, minimal error)
#
# This curve IS the training progress. Loss → 0 means the model learned.
# Try in Evaluate: loss_history[0], loss_history[-1], min(loss_history)


print(f"\n{'='*60}")
print("Session 6 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. Loss = single number measuring how wrong the prediction is
2. -log(pred) when truth=1: penalises confident wrong predictions HARD
3. Gradient of loss = direction to move prediction to reduce error
4. BCE (classification) vs MSE (regression) — different penalty curves
5. Total loss across examples = the error signal that drives training
6. Loss decreasing over epochs = the model is learning

Now open loss.cpp in CLion to see what happens INSIDE loss.backward()
— the chain rule computed step by step, gradients stored in memory.
""")
