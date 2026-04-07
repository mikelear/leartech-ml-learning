"""
Session 8: The Training Loop — Putting It All Together

Every piece assembled:
  features → forward → loss → backward → update → repeat

This is what real training code looks like.

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
# PART 1: Prepare the training data
# ============================================================

print("=" * 60)
print("PART 1: Training data")
print("=" * 60)

# More training examples than previous sessions
raw_data = [
    ("+ eval('code')\n+ innerHTML = data\n+ API_KEY = 'sk-123'", 1.0),
    ("+ const SECRET = 'password123'\n+ eval(input)", 1.0),
    ("+ innerHTML = userInput\n+ API_KEY = 'ghp_abc'", 1.0),
    ("+ document.innerHTML = response\n+ eval(json)", 1.0),
    ("+ const PASSWORD = 'admin'\n+ eval(data)", 1.0),
    ("+ import { HttpClient }\n+ constructor(private http: HttpClient)\n+ subscribe(", 0.0),
    ("+ import { DomSanitizer }\n+ constructor(private sanitizer: DomSanitizer)", 0.0),
    ("+ import { Observable }\n+ subscribe(data =>", 0.0),
    ("+ import { HttpClient }\n+ import { Observable }\n+ constructor(private http: HttpClient)", 0.0),
    ("+ import { DomSanitizer }\n+ import { HttpClient }\n+ subscribe(result =>", 0.0),
]

# Split: 8 for training, 2 for validation (unseen during training)
train_data = raw_data[:8]
val_data = raw_data[8:]

# Convert to tensors
train_features = torch.stack([extract_features(d) for d, _ in train_data])
train_labels = torch.tensor([[l] for _, l in train_data])
val_features = torch.stack([extract_features(d) for d, _ in val_data])
val_labels = torch.tensor([[l] for _, l in val_data])

print(f"\nTraining set: {train_features.shape[0]} examples")
print(f"Validation set: {val_features.shape[0]} examples")
print(f"Feature shape: {train_features.shape}")  # [8, 8] — 8 examples × 8 features
print(f"Label shape: {train_labels.shape}")       # [8, 1]
x = 1  # 🔴 BREAKPOINT — Line 71: inspect train_features, train_labels
# train_features is a BATCH — 8 examples stacked into a matrix.
# Shape [8, 8]: 8 examples, each with 8 features.
# The model processes them one at a time (or all at once as a batch).
# Try in Evaluate: train_features[0] (first example), train_labels[0] (its label)


# ============================================================
# PART 2: One complete training step (annotated)
# ============================================================

print(f"\n{'='*60}")
print("PART 2: One complete training step")
print(f"{'='*60}")

torch.manual_seed(42)
model = CodeReviewNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

# One example
features = train_features[0]  # First example
label = train_labels[0]       # Its label

# Step 1: Forward pass
prediction = model(features)
print(f"\n  1. Forward:  prediction = {prediction.item():.4f}")

# Step 2: Compute loss
loss = loss_fn(prediction, label)
print(f"  2. Loss:     {loss.item():.4f}")

# Step 3: Zero old gradients (IMPORTANT — they accumulate otherwise)
optimizer.zero_grad()
print(f"  3. Zero grad: cleared old gradients")

# Step 4: Backward pass (compute new gradients)
loss.backward()
print(f"  4. Backward: gradients computed")
print(f"     layer1.weight.grad mean: {model.layer1.weight.grad.abs().mean():.6f}")

# Step 5: Update weights
optimizer.step()
print(f"  5. Step:     weights updated")

# Check: prediction improved?
new_pred = model(features)
print(f"\n  Before: {prediction.item():.4f}  After: {new_pred.item():.4f}  Target: {label.item():.0f}")
x = 2  # 🔴 BREAKPOINT — Line 95: inspect model parameters before/after step
# This is THE training loop in its simplest form.
# Every ML training job on earth does these 5 steps.
# GPT-4 training: same 5 steps, just billions of parameters and trillions of examples.


# ============================================================
# PART 3: SGD vs Adam optimizer
# ============================================================

print(f"\n{'='*60}")
print("PART 3: SGD vs Adam")
print(f"{'='*60}")

def train_model(model, optimizer, epochs=100):
    """Train and return loss history."""
    loss_fn = nn.BCELoss()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(train_features)):
            optimizer.zero_grad()
            pred = model(train_features[i])
            loss = loss_fn(pred, train_labels[i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / len(train_features))
    return history

# SGD — simple: weight -= lr * gradient
torch.manual_seed(42)
sgd_model = CodeReviewNet()
sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=0.1)
sgd_history = train_model(sgd_model, sgd_optimizer)

# Adam — adaptive: adjusts learning rate per-parameter based on past gradients
torch.manual_seed(42)
adam_model = CodeReviewNet()
adam_optimizer = torch.optim.Adam(adam_model.parameters(), lr=0.01)
adam_history = train_model(adam_model, adam_optimizer)

print(f"\nFinal loss after 100 epochs:")
print(f"  SGD  (lr=0.1):  {sgd_history[-1]:.6f}")
print(f"  Adam (lr=0.01): {adam_history[-1]:.6f}")
print(f"\nLoss at epoch 10:")
print(f"  SGD:  {sgd_history[9]:.6f}")
print(f"  Adam: {adam_history[9]:.6f}")
print(f"  Adam converges faster — it adapts the learning rate per-weight")
x = 3  # 🔴 BREAKPOINT — Line 121: inspect sgd_history, adam_history
# Adam typically converges faster because it:
#   - Keeps a running average of past gradients (momentum)
#   - Keeps a running average of gradient magnitude (adaptive learning rate)
#   - Weights that need big updates get big steps
#   - Weights that are already good get small steps
# SGD treats every weight the same — flat learning rate for all.
# Try in Evaluate: sgd_history[0], adam_history[0] (starting loss)


# ============================================================
# PART 4: Learning rate — the most important hyperparameter
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Learning rate effect")
print(f"{'='*60}")

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
lr_histories = {}

for lr in learning_rates:
    torch.manual_seed(42)
    m = CodeReviewNet()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    history = train_model(m, opt, epochs=50)
    lr_histories[lr] = history

    final = history[-1]
    early = history[4]
    status = "GOOD" if final < 0.1 else ("SLOW" if final < 0.5 else "BAD")
    if any(h != h for h in history):  # NaN check
        status = "EXPLODED"
        final = float('nan')
    print(f"  lr={lr:<5.3f}  loss@5={early:>8.4f}  loss@50={final:>8.4f}  {status}")

x = 4  # 🔴 BREAKPOINT — Line 155: inspect lr_histories
# The pattern:
#   lr=0.001: too slow — loss barely moves in 50 epochs
#   lr=0.01:  slow but steady
#   lr=0.1:   good — converges well
#   lr=0.5:   fast but risky — might overshoot
#   lr=1.0:   unstable — loss oscillates
#   lr=5.0:   EXPLODES — loss goes to infinity (NaN)
#
# The learning rate is the MOST important hyperparameter.
# Too small = wastes time. Too large = never converges. Just right = fast + stable.
# Try in Evaluate: lr_histories[0.1][-1], lr_histories[5.0][0]


# ============================================================
# PART 5: Learning rate scheduling
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Learning rate scheduling")
print(f"{'='*60}")

# Start with a high learning rate, reduce over time
# Like taking big steps when far from the target, small steps when close

torch.manual_seed(42)
model = CodeReviewNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
# Reduce LR by 50% every 20 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

scheduled_history = []
lr_at_epoch = []

loss_fn = nn.BCELoss()
for epoch in range(100):
    total_loss = 0
    for i in range(len(train_features)):
        optimizer.zero_grad()
        pred = model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduled_history.append(total_loss / len(train_features))
    lr_at_epoch.append(optimizer.param_groups[0]['lr'])
    scheduler.step()  # Adjust learning rate

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={scheduled_history[-1]:.6f}  lr={lr_at_epoch[-1]:.4f}")

x = 5  # 🔴 BREAKPOINT — Line 191: inspect scheduled_history, lr_at_epoch
# Learning rate decreases over time:
#   Epoch 0-19:  lr=0.500 (big steps, fast learning)
#   Epoch 20-39: lr=0.250 (medium steps)
#   Epoch 40-59: lr=0.125 (small steps, fine-tuning)
#   Epoch 60+:   lr=0.063 (tiny steps, polishing)
# This combines fast early learning with stable late convergence.


# ============================================================
# PART 6: Full training with validation
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Training with validation (the real deal)")
print(f"{'='*60}")

torch.manual_seed(42)
model = CodeReviewNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

train_history = []
val_history = []

print(f"\n{'Epoch':>6s}  {'Train Loss':>12s}  {'Val Loss':>12s}  {'Train Acc':>10s}  {'Val Acc':>10s}")
print(f"{'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")

for epoch in range(200):
    # --- Training ---
    model.train()  # Enable training mode (affects dropout, batch norm)
    total_train_loss = 0
    train_correct = 0
    for i in range(len(train_features)):
        optimizer.zero_grad()
        pred = model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_correct += ((pred > 0.5).float() == train_labels[i]).item()

    train_loss = total_train_loss / len(train_features)
    train_acc = train_correct / len(train_features)
    train_history.append(train_loss)

    # --- Validation (NO gradient computation, NO weight updates) ---
    model.eval()   # Disable training mode
    total_val_loss = 0
    val_correct = 0
    with torch.no_grad():  # Saves memory and computation
        for i in range(len(val_features)):
            pred = model(val_features[i])
            loss = loss_fn(pred, val_labels[i])
            total_val_loss += loss.item()
            val_correct += ((pred > 0.5).float() == val_labels[i]).item()

    val_loss = total_val_loss / len(val_features)
    val_acc = val_correct / len(val_features)
    val_history.append(val_loss)

    if epoch % 20 == 0:
        print(f"  {epoch:>4d}  {train_loss:>12.6f}  {val_loss:>12.6f}  {train_acc:>9.0%}  {val_acc:>9.0%}")

x = 6  # 🔴 BREAKPOINT — Line 223: inspect train_history, val_history
# Two loss curves:
#   train_loss: should always decrease (model fits training data)
#   val_loss: should decrease then plateau (model generalises)
#
# If val_loss starts INCREASING while train_loss decreases → OVERFITTING
# (the model memorised training data instead of learning general patterns)
# We'll explore this in Session 9.
#
# model.train() vs model.eval() — some layers behave differently:
#   Dropout: active in train (randomly zeros neurons), inactive in eval
#   BatchNorm: uses running stats in eval, batch stats in train
# Our simple network doesn't use these, but it's good practice.


# ============================================================
# PART 7: Final model — what did it learn?
# ============================================================

print(f"\n{'='*60}")
print("PART 7: The trained model")
print(f"{'='*60}")

model.eval()
print("\nAll predictions:")

all_data = raw_data
for diff, truth in all_data:
    features = extract_features(diff)
    with torch.no_grad():
        pred = model(features)
    verdict = "FAIL" if pred.item() > 0.5 else "PASS"
    correct = "✓" if (pred.item() > 0.5) == (truth == 1.0) else "✗"
    source = "train" if (diff, truth) in train_data else "val"
    print(f"  {correct} {verdict} ({pred.item():.3f}) truth={truth:.0f} [{source}]  {diff[:50]}...")

# Count correct
with torch.no_grad():
    train_preds = model(train_features)
    val_preds = model(val_features)
    train_acc = ((train_preds > 0.5).float() == train_labels).float().mean()
    val_acc = ((val_preds > 0.5).float() == val_labels).float().mean()

print(f"\nFinal accuracy:")
print(f"  Training:   {train_acc.item():.0%}")
print(f"  Validation: {val_acc.item():.0%}")

x = 7  # 🔴 BREAKPOINT — Line 260: inspect train_acc, val_acc, model
# The model should get 100% on training data (it saw these examples)
# Validation accuracy tells you if it GENERALISED or just memorised.
# With only 2 val examples it's hard to tell — Session 10 uses real data.


print(f"\n{'='*60}")
print("Session 8 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. Training loop: forward → loss → zero_grad → backward → step → repeat
2. Epoch = one pass through ALL training data
3. Adam adapts learning rate per-weight (usually better than SGD)
4. Learning rate: too small = slow, too large = explodes, just right = fast + stable
5. LR scheduling: start big, shrink over time (fast start, stable finish)
6. Train/validation split: train_loss shows fitting, val_loss shows generalisation
7. model.train() vs model.eval() — important for dropout/batch norm layers

The complete ML pipeline is now understood:
  Data → Features → Model → Forward → Loss → Backward → Update → Repeat

  Then evaluate on UNSEEN data to check if it actually learned.

Next session: Overfitting — when the model memorises instead of learning
""")
