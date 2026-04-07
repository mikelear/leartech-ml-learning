"""
Session 9: Overfitting — When the Model Memorises Instead of Learning

A model that gets 100% on training data but fails on new data is USELESS.
This session shows how to detect and prevent it.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import re
import random


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
# Generate more data with noise (realistic)
# ============================================================

def generate_data(n_samples=50):
    """Generate training data with some noise — like real-world data."""
    bad_templates = [
        "+ eval('{noise}')\n+ API_KEY = 'sk-{noise}'",
        "+ innerHTML = {noise}\n+ const SECRET = '{noise}'",
        "+ eval(input)\n+ document.innerHTML = data",
        "+ const PASSWORD = '{noise}'\n+ eval({noise})",
    ]
    good_templates = [
        "+ import {{ HttpClient }}\n+ constructor(private http: HttpClient)\n+ subscribe(",
        "+ import {{ DomSanitizer }}\n+ constructor(private sanitizer: DomSanitizer)",
        "+ import {{ Observable }}\n+ subscribe(data =>",
        "+ import {{ HttpClient }}\n+ import {{ Observable }}",
    ]

    data = []
    for _ in range(n_samples):
        if random.random() > 0.5:
            template = random.choice(bad_templates)
            diff = template.replace("{noise}", str(random.randint(100, 999)))
            # Add noise: sometimes bad code also has imports
            if random.random() > 0.7:
                diff += "\n+ import { Component }"
            data.append((diff, 1.0))
        else:
            template = random.choice(good_templates)
            diff = template.replace("{noise}", str(random.randint(100, 999)))
            data.append((diff, 0.0))

    random.shuffle(data)
    features = torch.stack([extract_features(d) for d, _ in data])
    labels = torch.tensor([[l] for _, l in data])
    return features, labels


random.seed(42)
torch.manual_seed(42)

# Small dataset — easy to overfit
train_features, train_labels = generate_data(20)
val_features, val_labels = generate_data(30)

print("=" * 60)
print("Data: 20 training examples, 30 validation examples")
print("=" * 60)


# ============================================================
# PART 1: A model that WILL overfit
# ============================================================

class OverfitNet(nn.Module):
    """Deliberately too big — 512 hidden neurons for 20 training examples."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 512)   # Way too wide!
        self.layer2 = nn.Linear(512, 256)  # Even wider!
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x))


print(f"\n{'='*60}")
print("PART 1: Training an OVERFIT model")
print(f"{'='*60}")

model = OverfitNet()
param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel has {param_count:,} parameters for only {len(train_features)} examples!")
print(f"Ratio: {param_count / len(train_features):.0f} parameters per training example")
print(f"(Should be < 10 per example for good generalisation)")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

train_losses = []
val_losses = []

x = 1  # 🔴 BREAKPOINT — Line 95: inspect param_count, model
# 135,425 parameters for 20 training examples!
# That's 6,771 parameters per example — the model can easily memorise everything.

for epoch in range(300):
    # Train
    model.train()
    total_train = 0
    for i in range(len(train_features)):
        optimizer.zero_grad()
        pred = model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        optimizer.step()
        total_train += loss.item()
    train_losses.append(total_train / len(train_features))

    # Validate
    model.eval()
    with torch.no_grad():
        total_val = 0
        for i in range(len(val_features)):
            pred = model(val_features[i])
            loss = loss_fn(pred, val_labels[i])
            total_val += loss.item()
        val_losses.append(total_val / len(val_features))

    if epoch % 30 == 0:
        print(f"  Epoch {epoch:3d}: train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}")

x = 2  # 🔴 BREAKPOINT — Line 120: inspect train_losses, val_losses
# THE OVERFIT SIGNATURE:
#   train_loss → 0 (model memorised training data perfectly)
#   val_loss → increases (model FAILS on new data)
#
# Try in Evaluate:
#   train_losses[-1]  (near 0 — perfect on training)
#   val_losses[-1]    (higher — poor on validation)
#   min(val_losses)   (when was validation BEST? Early in training!)
#   val_losses.index(min(val_losses))  (the epoch to stop at)

# Find the overfit point
best_val_epoch = val_losses.index(min(val_losses))
print(f"\nOverfit analysis:")
print(f"  Train loss: {train_losses[-1]:.6f} (near 0 = memorised)")
print(f"  Val loss:   {val_losses[-1]:.6f} (getting worse)")
print(f"  Best val was at epoch {best_val_epoch} (should have stopped here!)")


# ============================================================
# PART 2: Fix 1 — Dropout
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Fix with Dropout")
print(f"{'='*60}")

class DropoutNet(nn.Module):
    """Same architecture but with dropout — randomly disables neurons."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 512)
        self.drop1 = nn.Dropout(0.5)     # Kill 50% of neurons randomly each forward pass
        self.layer2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.drop1(torch.relu(self.layer1(x)))
        x = self.drop2(torch.relu(self.layer2(x)))
        return torch.sigmoid(self.layer3(x))


torch.manual_seed(42)
dropout_model = DropoutNet()
dropout_opt = torch.optim.Adam(dropout_model.parameters(), lr=0.001)

dropout_train_losses = []
dropout_val_losses = []

for epoch in range(300):
    dropout_model.train()  # Dropout ACTIVE
    total_train = 0
    for i in range(len(train_features)):
        dropout_opt.zero_grad()
        pred = dropout_model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        dropout_opt.step()
        total_train += loss.item()
    dropout_train_losses.append(total_train / len(train_features))

    dropout_model.eval()  # Dropout INACTIVE (use all neurons for prediction)
    with torch.no_grad():
        total_val = 0
        for i in range(len(val_features)):
            pred = dropout_model(val_features[i])
            loss = loss_fn(pred, val_labels[i])
            total_val += loss.item()
        dropout_val_losses.append(total_val / len(val_features))

x = 3  # 🔴 BREAKPOINT — Line 155: inspect dropout_val_losses vs val_losses
# Compare:
#   val_losses[-1]          (overfit model — high, getting worse)
#   dropout_val_losses[-1]  (dropout model — should be lower/more stable)
#
# Dropout works by forcing the network to NOT rely on any single neuron.
# If neuron 5 memorises "example 3 is FAIL", dropout randomly kills neuron 5,
# so the network must learn the PATTERN, not the specific example.
print(f"\nDropout effect:")
print(f"  Without dropout: final val_loss = {val_losses[-1]:.4f}")
print(f"  With dropout:    final val_loss = {dropout_val_losses[-1]:.4f}")


# ============================================================
# PART 3: Fix 2 — Early Stopping
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Fix with Early Stopping")
print(f"{'='*60}")

torch.manual_seed(42)
es_model = OverfitNet()  # Same overfit-prone architecture
es_opt = torch.optim.Adam(es_model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 20       # Stop if val_loss doesn't improve for 20 epochs
patience_counter = 0
best_weights = None
stopped_at = 300

for epoch in range(300):
    es_model.train()
    for i in range(len(train_features)):
        es_opt.zero_grad()
        pred = es_model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        es_opt.step()

    es_model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(es_model(val_features[i]), val_labels[i]).item()
                       for i in range(len(val_features))) / len(val_features)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.clone() for k, v in es_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        stopped_at = epoch
        print(f"  Early stopping at epoch {epoch}! Val loss hasn't improved for {patience} epochs.")
        break

    if epoch % 30 == 0:
        print(f"  Epoch {epoch:3d}: val_loss={val_loss:.4f}  patience={patience_counter}/{patience}")

# Restore best weights
es_model.load_state_dict(best_weights)

x = 4  # 🔴 BREAKPOINT — Line 202: inspect stopped_at, best_val_loss
# Early stopping caught the overfit!
# Instead of training for 300 epochs (memorising), it stopped when
# validation stopped improving — typically around epoch 30-60.
print(f"\nEarly stopping result:")
print(f"  Stopped at epoch: {stopped_at}")
print(f"  Best val_loss: {best_val_loss:.4f}")
print(f"  Overfit model val_loss (epoch 300): {val_losses[-1]:.4f}")


# ============================================================
# PART 4: Fix 3 — Weight Decay (L2 Regularisation)
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Fix with Weight Decay")
print(f"{'='*60}")

# Weight decay adds a penalty for large weights:
# actual_loss = prediction_loss + weight_decay * sum(weight²)
# This prevents any single weight from becoming too large (memorising)

torch.manual_seed(42)
wd_model = OverfitNet()
# weight_decay=0.01 adds L2 penalty — same as adding 0.01 * sum(w²) to the loss
wd_opt = torch.optim.Adam(wd_model.parameters(), lr=0.001, weight_decay=0.01)

wd_val_losses = []
for epoch in range(300):
    wd_model.train()
    for i in range(len(train_features)):
        wd_opt.zero_grad()
        pred = wd_model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        wd_opt.step()

    wd_model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(wd_model(val_features[i]), val_labels[i]).item()
                       for i in range(len(val_features))) / len(val_features)
        wd_val_losses.append(val_loss)

x = 5  # 🔴 BREAKPOINT — Line 234: inspect wd_val_losses
# Weight decay keeps weights small → model can't overfit as easily.
# Compare weight magnitudes:
print(f"\nWeight magnitudes (average abs value):")
print(f"  Overfit model:     {sum(p.abs().mean().item() for p in model.parameters()) / len(list(model.parameters())):.4f}")
print(f"  Weight decay model: {sum(p.abs().mean().item() for p in wd_model.parameters()) / len(list(wd_model.parameters())):.4f}")
print(f"\nVal loss comparison:")
print(f"  Weight decay: {wd_val_losses[-1]:.4f}")


# ============================================================
# PART 5: Fix 4 — Simpler model (the best fix)
# ============================================================

print(f"\n{'='*60}")
print("PART 5: All fixes compared")
print(f"{'='*60}")

class SimpleNet(nn.Module):
    """Right-sized model — 16 hidden neurons, not 512."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer2(torch.relu(self.layer1(x))))


torch.manual_seed(42)
simple_model = SimpleNet()
simple_opt = torch.optim.Adam(simple_model.parameters(), lr=0.01)

simple_val_losses = []
for epoch in range(300):
    simple_model.train()
    for i in range(len(train_features)):
        simple_opt.zero_grad()
        pred = simple_model(train_features[i])
        loss = loss_fn(pred, train_labels[i])
        loss.backward()
        simple_opt.step()

    simple_model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(simple_model(val_features[i]), val_labels[i]).item()
                       for i in range(len(val_features))) / len(val_features)
        simple_val_losses.append(val_loss)

simple_params = sum(p.numel() for p in simple_model.parameters())

print(f"\n{'Model':>20s}  {'Params':>8s}  {'Val Loss':>10s}")
print(f"{'─'*20}  {'─'*8}  {'─'*10}")
print(f"{'Overfit (512 hidden)':>20s}  {param_count:>8,}  {val_losses[-1]:>10.4f}")
print(f"{'+ Dropout':>20s}  {param_count:>8,}  {dropout_val_losses[-1]:>10.4f}")
print(f"{'+ Early Stopping':>20s}  {param_count:>8,}  {best_val_loss:>10.4f}")
print(f"{'+ Weight Decay':>20s}  {param_count:>8,}  {wd_val_losses[-1]:>10.4f}")
print(f"{'Simple (16 hidden)':>20s}  {simple_params:>8,}  {simple_val_losses[-1]:>10.4f}")

x = 6  # 🔴 BREAKPOINT — Line 278: compare all approaches
# Usually the simple model wins:
#   - Fewer parameters = less capacity to memorise
#   - More likely to learn GENERAL patterns
#   - Faster to train, easier to debug
#
# The rule: start simple, add complexity only if you need it.
# "135,000 parameters for 20 examples" was the real problem.
# "145 parameters for 20 examples" is more reasonable.


print(f"\n{'='*60}")
print("Session 9 Complete!")
print(f"{'='*60}")
print("""
Key concepts:
1. Overfitting = memorising training data instead of learning patterns
2. Detected by: train_loss ↓ while val_loss ↑ (the gap grows)
3. Fixes:
   - Dropout: randomly kill neurons → forces redundancy
   - Early stopping: stop when val_loss stops improving
   - Weight decay: penalise large weights → simpler internal model
   - Simpler architecture: fewer parameters = less memorisation capacity
4. The best fix is usually: RIGHT-SIZE the model for your data
5. Rule of thumb: parameters should be < 10× training examples

For our classifier (Session 10):
  - ~500 feedback examples → model should have < 5,000 parameters
  - That's a 2-layer network with 16-32 hidden neurons
  - Dropout + early stopping as insurance

Next session: Our Classifier — building the real PASS/FAIL model on leartech data
""")
