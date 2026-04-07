"""
Session 10: Our Classifier — PASS/FAIL on Real Code Review Data

Everything from Sessions 1-9 assembled into a REAL model
trained on REAL data from the leartech AI review pipeline.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import torch.nn as nn
import json
import os
import re
from pathlib import Path


# ============================================================
# PART 1: Load real feedback data
# ============================================================

print("=" * 60)
print("PART 1: Loading real feedback data")
print("=" * 60)

# Path to the feedback data — adjust if needed
FEEDBACK_DIR = "/tmp/leartech-llm-training-data/feedback"

# If not cloned, try local path
if not os.path.exists(FEEDBACK_DIR):
    FEEDBACK_DIR = os.path.expanduser("~/leartech/leartech-llm-training-data/feedback")

def load_feedback_data(feedback_dir: str) -> list[dict]:
    """Load all feedback JSON files."""
    data = []
    for json_file in Path(feedback_dir).rglob("*.json"):
        try:
            with open(json_file) as f:
                record = json.load(f)
                if "diff" in record and "overall_verdict" in record:
                    data.append(record)
        except (json.JSONDecodeError, KeyError):
            continue
    return data


raw_data = load_feedback_data(FEEDBACK_DIR)
print(f"\nLoaded {len(raw_data)} feedback records")

# Show the data distribution
verdicts = [d["overall_verdict"] for d in raw_data]
scores = [d.get("individual_reviews", [{}])[0].get("score", 0) for d in raw_data if d.get("individual_reviews")]
pass_count = verdicts.count("PASS")
fail_count = len(verdicts) - pass_count

print(f"  PASS: {pass_count}")
print(f"  FAIL/WARN/ERROR: {fail_count}")
if scores:
    print(f"  Score range: {min(scores)} - {max(scores)}")
    print(f"  Mean score: {sum(scores)/len(scores):.1f}")

# Convert verdict to binary label
# PASS = 0 (good code), everything else = 1 (bad code)
labels = [0.0 if d["overall_verdict"] == "PASS" else 1.0 for d in raw_data]
diffs = [d["diff"] for d in raw_data]

x = 1  # 🔴 BREAKPOINT — Line 72: inspect raw_data, labels, pass_count, fail_count
# Look at raw_data[0] — a real feedback record with diff, verdict, issues
# labels is a list of 0s and 1s
# Check the balance: if 90% PASS and 10% FAIL, the model might
# just predict PASS every time and get 90% accuracy (useless!)


# ============================================================
# PART 2: Extract features from real diffs
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Feature extraction from real diffs")
print(f"{'='*60}")

def extract_features(diff: str) -> torch.Tensor:
    """Extract features from a real code diff."""
    features = [
        # Security signals
        len(re.findall(r'eval\s*\(', diff)),
        len(re.findall(r'innerHTML', diff)),
        len(re.findall(r'(API_KEY|SECRET|PASSWORD|TOKEN)', diff, re.IGNORECASE)),
        len(re.findall(r'(sk-|ghp_|password|secret)', diff, re.IGNORECASE)),

        # Code quality signals
        len(re.findall(r'^[\+].*import\s+', diff, re.MULTILINE)),
        len(re.findall(r'constructor', diff)),
        len(re.findall(r'(subscribe|Observable|Promise)', diff)),
        len(re.findall(r'(HttpClient|DomSanitizer|Injectable)', diff)),

        # Diff metrics
        len(re.findall(r'^\+', diff, re.MULTILINE)),     # lines added
        len(re.findall(r'^-', diff, re.MULTILINE)),       # lines removed
        len(diff.split('\n')),                              # total lines
        len(re.findall(r'(function|func |def |=>)', diff)), # function count
        len(re.findall(r'(if |else|switch|case)', diff)),   # control flow
        len(re.findall(r'(try|catch|error|Error)', diff)),  # error handling
        len(re.findall(r'(test|spec|Test|Spec)', diff)),    # test-related
        len(re.findall(r'(TODO|FIXME|HACK|XXX)', diff)),    # code debt markers
    ]
    return torch.tensor(features, dtype=torch.float32)


FEATURE_NAMES = [
    "eval_calls", "innerHTML", "secret_names", "secret_patterns",
    "imports", "constructor", "async_patterns", "angular_services",
    "lines_added", "lines_removed", "total_lines", "functions",
    "control_flow", "error_handling", "test_related", "code_debt"
]

# Extract features for all diffs
all_features = torch.stack([extract_features(d) for d in diffs])
all_labels = torch.tensor(labels).unsqueeze(1)

print(f"\nFeature matrix shape: {all_features.shape}")  # [N, 16]
print(f"Labels shape: {all_labels.shape}")                # [N, 1]

# Feature statistics
print(f"\nFeature means (across all examples):")
means = all_features.mean(dim=0)
for name, mean in zip(FEATURE_NAMES, means):
    bar = "█" * int(mean.item() * 2)
    print(f"  {name:20s}: {mean.item():.2f}  {bar}")

x = 2  # 🔴 BREAKPOINT — Line 107: inspect all_features, all_labels
# all_features is [N, 16] — N examples, 16 features each
# Some features will be mostly 0 (eval, innerHTML — rare in real code)
# Others will be common (lines_added, imports)
# Try in Evaluate: all_features[:, 0].sum() (total eval calls across all diffs)


# ============================================================
# PART 3: Train/validation/test split
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Data split")
print(f"{'='*60}")

# Shuffle the data
n = len(all_features)
indices = torch.randperm(n)
all_features = all_features[indices]
all_labels = all_labels[indices]

# 70% train, 15% validation, 15% test
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_X, train_y = all_features[:train_end], all_labels[:train_end]
val_X, val_y = all_features[train_end:val_end], all_labels[train_end:val_end]
test_X, test_y = all_features[val_end:], all_labels[val_end:]

print(f"\n  Training:   {len(train_X)} examples")
print(f"  Validation: {len(val_X)} examples")
print(f"  Test:       {len(test_X)} examples")
print(f"\n  Train FAIL ratio: {train_y.mean().item():.1%}")
print(f"  Val FAIL ratio:   {val_y.mean().item():.1%}")
print(f"  Test FAIL ratio:  {test_y.mean().item():.1%}")

x = 3  # 🔴 BREAKPOINT — Line 130: inspect train_X, val_X, test_X shapes
# Three separate datasets:
#   train: model SEES this during training
#   val: model DOESN'T see this — used to detect overfitting
#   test: model NEVER sees this — final evaluation only


# ============================================================
# PART 4: Build the right-sized model
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Model architecture")
print(f"{'='*60}")

NUM_FEATURES = 16

class CodeClassifier(nn.Module):
    """Right-sized for ~100 examples: small, with dropout."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = CodeClassifier()
param_count = sum(p.numel() for p in model.parameters())
print(f"\n  Model: {param_count} parameters")
print(f"  Training examples: {len(train_X)}")
print(f"  Ratio: {param_count / len(train_X):.1f} params per example")
print(f"  (Under 10 is good — we're at {param_count / len(train_X):.1f})")

x = 4  # 🔴 BREAKPOINT — Line 158: inspect model, param_count
# ~1,100 parameters for ~70 training examples ≈ 16 per example
# Slightly high but dropout will help prevent overfitting
# Architecture: [16] → [32] → dropout → [16] → dropout → [1]


# ============================================================
# PART 5: Train with early stopping
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Training")
print(f"{'='*60}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

train_history = []
val_history = []
best_val_loss = float('inf')
best_weights = None
patience = 30
patience_counter = 0

for epoch in range(500):
    # Train
    model.train()
    total_train_loss = 0
    for i in range(len(train_X)):
        optimizer.zero_grad()
        pred = model(train_X[i])
        loss = loss_fn(pred, train_y[i])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_history.append(total_train_loss / len(train_X))

    # Validate
    model.eval()
    with torch.no_grad():
        val_preds = model(val_X)
        val_loss = loss_fn(val_preds, val_y).item()
    val_history.append(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 50 == 0:
        train_acc = ((model(train_X) > 0.5).float() == train_y).float().mean()
        val_acc = ((val_preds > 0.5).float() == val_y).float().mean()
        print(f"  Epoch {epoch:3d}: train_loss={train_history[-1]:.4f}  "
              f"val_loss={val_loss:.4f}  train_acc={train_acc:.0%}  val_acc={val_acc:.0%}")

    if patience_counter >= patience:
        print(f"\n  Early stopping at epoch {epoch}!")
        break

# Restore best weights
model.load_state_dict(best_weights)
model.eval()

x = 5  # 🔴 BREAKPOINT — Line 193: inspect train_history, val_history, epoch
# Watch: train_loss ↓, val_loss ↓ (then maybe ↑ → early stopping kicks in)
# Try in Evaluate: min(val_history), val_history.index(min(val_history))


# ============================================================
# PART 6: Evaluate — precision, recall, F1
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Evaluation metrics")
print(f"{'='*60}")

with torch.no_grad():
    test_preds = model(test_X)
    test_pred_labels = (test_preds > 0.5).float()

# Confusion matrix
TP = ((test_pred_labels == 1) & (test_y == 1)).sum().item()  # True Positive
TN = ((test_pred_labels == 0) & (test_y == 0)).sum().item()  # True Negative
FP = ((test_pred_labels == 1) & (test_y == 0)).sum().item()  # False Positive
FN = ((test_pred_labels == 0) & (test_y == 1)).sum().item()  # False Negative

accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0      # Of predicted FAIL, how many actually FAIL?
recall = TP / (TP + FN) if (TP + FN) > 0 else 0          # Of actual FAIL, how many did we catch?
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  PASS    FAIL")
print(f"  Actual PASS  [  {TN:3.0f}     {FP:3.0f}  ]")
print(f"  Actual FAIL  [  {FN:3.0f}     {TP:3.0f}  ]")

print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy:.1%}  (overall correct)")
print(f"  Precision: {precision:.1%}  (of predicted FAIL, how many are actually FAIL)")
print(f"  Recall:    {recall:.1%}  (of actual FAIL, how many did we catch)")
print(f"  F1 Score:  {f1:.1%}  (harmonic mean of precision and recall)")

x = 6  # 🔴 BREAKPOINT — Line 222: inspect TP, TN, FP, FN, accuracy, precision, recall, f1
# The confusion matrix tells the full story:
#   TP: correctly identified bad code (good!)
#   TN: correctly identified good code (good!)
#   FP: flagged good code as bad (annoying but safe)
#   FN: missed bad code (dangerous!)
#
# For a code review classifier:
#   High precision = fewer false alarms
#   High recall = catches more bad code (more important for security!)


# ============================================================
# PART 7: Look at individual predictions
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Individual test predictions")
print(f"{'='*60}")

with torch.no_grad():
    for i in range(len(test_X)):
        pred = model(test_X[i]).item()
        truth = test_y[i].item()
        verdict = "FAIL" if pred > 0.5 else "PASS"
        actual = "FAIL" if truth == 1.0 else "PASS"
        correct = "✓" if verdict == actual else "✗"
        confidence = pred if pred > 0.5 else 1 - pred

        # Find the original diff (approximate — we shuffled)
        features = test_X[i]
        has_eval = features[0].item() > 0
        has_secrets = features[2].item() > 0
        lines = features[10].item()
        flags = []
        if has_eval: flags.append("eval")
        if has_secrets: flags.append("secrets")
        flag_str = ", ".join(flags) if flags else "clean"

        print(f"  {correct} pred={verdict}({pred:.3f}) actual={actual}  "
              f"conf={confidence:.1%}  [{flag_str}, {lines:.0f} lines]")

x = 7  # 🔴 BREAKPOINT — Line 258: inspect individual predictions
# Look for patterns in the mistakes:
#   - Does it miss subtle bad code? (FN)
#   - Does it flag clean code incorrectly? (FP)
#   - Are high-confidence predictions always correct?


# ============================================================
# PART 8: Save the model
# ============================================================

print(f"\n{'='*60}")
print("PART 8: Save the trained model")
print(f"{'='*60}")

model_path = os.path.join(os.path.dirname(__file__), "code_classifier.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_names': FEATURE_NAMES,
    'num_features': NUM_FEATURES,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'training_examples': len(train_X),
    'epochs_trained': epoch + 1,
}, model_path)

print(f"\n  Model saved to: {model_path}")
print(f"  Size: {os.path.getsize(model_path) / 1024:.1f} KB")

x = 8  # 🔴 BREAKPOINT — Line 282: inspect the saved model info
# The .pt file contains:
#   - All weight tensors (the trained model)
#   - Metadata (accuracy, feature names, etc.)
#   - This file IS the model — deploy it, load it, make predictions
#
# Compare size:
#   Our model: ~5 KB
#   Qwen 14B: ~8 GB
#   Same concept, wildly different scale.


print(f"\n{'='*60}")
print("Session 10 Complete!")
print(f"{'='*60}")
print(f"""
Results:
  Accuracy:  {accuracy:.1%}
  Precision: {precision:.1%}
  Recall:    {recall:.1%}
  F1 Score:  {f1:.1%}
  Model size: {os.path.getsize(model_path) / 1024:.1f} KB

Key concepts:
1. Real data has noise, imbalance, and edge cases — not like toy examples
2. 16 hand-crafted features from code diffs (security + quality + metrics)
3. Right-sized model with dropout + early stopping prevents overfitting
4. Confusion matrix: TP/TN/FP/FN tells the full story, not just accuracy
5. For security: recall > precision (missing bad code is worse than false alarms)
6. Model saved as .pt file — ready for deployment in Session 11

Next session: Deploy it — Flask API + Kubernetes deployment + integrate with pipeline
""")
