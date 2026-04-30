"""
Session 10.5: Better Features — TF-IDF + Embeddings

Session 10's classifier uses 16 hand-crafted regex features and sits at 43%
accuracy on the eval suite. The PASS cases all cluster at 0.52–0.57 probability
— the model can't tell them apart from FAIL.

This session teaches THREE levels of feature extraction:
  1. Hand-crafted regex (Session 10 — what we have)
  2. TF-IDF (automatic — finds distinctive words)
  3. Feature stacking (combine all of the above)

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.

In production: leartech-ai-classifier/app/features.py would gain tfidf_features()
alongside the existing extract_features().
"""

import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ============================================================
# PART 1: Load real feedback data (same as Session 10)
# ============================================================

print("=" * 60)
print("PART 1: Loading real feedback data")
print("=" * 60)

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
diffs = [d["diff"] for d in raw_data]
labels = [0.0 if d["overall_verdict"] == "PASS" else 1.0 for d in raw_data]

pass_count = labels.count(0.0)
fail_count = labels.count(1.0)

print(f"\nLoaded {len(raw_data)} feedback records")
print(f"  PASS: {pass_count}")
print(f"  FAIL/WARN/ERROR: {fail_count}")
print(f"  Balance: {pass_count / len(raw_data):.0%} PASS / {fail_count / len(raw_data):.0%} FAIL")

x = 1  # 🔴 BREAKPOINT — Line 62: inspect raw_data, labels
# Check the class balance. If very imbalanced the model may just
# predict the majority class every time and look "accurate".
# We'll handle this with class weights later.


# ============================================================
# PART 2: Session 10 features (baseline — what we're improving)
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Session 10 baseline — 16 hand-crafted features")
print(f"{'='*60}")

FEATURE_NAMES_V1 = [
    "eval_calls", "innerHTML", "secret_names", "secret_patterns",
    "imports", "constructor", "async_patterns", "angular_services",
    "lines_added", "lines_removed", "total_lines", "functions",
    "control_flow", "error_handling", "test_related", "code_debt"
]


def extract_features_v1(diff: str) -> list[float]:
    """Session 10 feature extractor — 16 regex counts."""
    return [
        len(re.findall(r'eval\s*\(', diff)),
        len(re.findall(r'innerHTML', diff)),
        len(re.findall(r'(API_KEY|SECRET|PASSWORD|TOKEN)', diff, re.IGNORECASE)),
        len(re.findall(r'(sk-|ghp_|password|secret)', diff, re.IGNORECASE)),
        len(re.findall(r'^[\+].*import\s+', diff, re.MULTILINE)),
        len(re.findall(r'constructor', diff)),
        len(re.findall(r'(subscribe|Observable|Promise)', diff)),
        len(re.findall(r'(HttpClient|DomSanitizer|Injectable)', diff)),
        len(re.findall(r'^\+', diff, re.MULTILINE)),
        len(re.findall(r'^-', diff, re.MULTILINE)),
        len(diff.split('\n')),
        len(re.findall(r'(function|func |def |=>)', diff)),
        len(re.findall(r'(if |else|switch|case)', diff)),
        len(re.findall(r'(try|catch|error|Error)', diff)),
        len(re.findall(r'(test|spec|Test|Spec)', diff)),
        len(re.findall(r'(TODO|FIXME|HACK|XXX)', diff)),
    ]


features_v1 = np.array([extract_features_v1(d) for d in diffs])
labels_arr = np.array(labels)

print(f"\n  Feature matrix: {features_v1.shape}")  # [N, 16]

# Show which features actually vary
stds = features_v1.std(axis=0)
for name, std in sorted(zip(FEATURE_NAMES_V1, stds), key=lambda x: x[1]):
    bar = "█" * int(std * 2)
    print(f"  {name:20s}: std={std:.2f}  {bar}")

print(f"\n  Features with zero variance (useless):")
zero_var = [name for name, s in zip(FEATURE_NAMES_V1, stds) if s < 0.01]
print(f"    {zero_var if zero_var else 'None'}")

x = 2  # 🔴 BREAKPOINT — Line 116: inspect features_v1, stds
# Notice: many features have near-zero standard deviation — they're the same
# for almost every diff. eval_calls, innerHTML, secret_names are usually 0.
# The model can't learn from features that don't vary!
#
# This is WHY 16 regex features give 43% accuracy — most features are dead.
# lines_added, total_lines, imports probably carry all the signal.


# ============================================================
# PART 3: TF-IDF — let the algorithm find distinctive words
# ============================================================

print(f"\n{'='*60}")
print("PART 3: TF-IDF — automatic feature discovery")
print(f"{'='*60}")

print("""
  TF-IDF = Term Frequency × Inverse Document Frequency

  TF:  how often a word appears in THIS diff
  IDF: how rare the word is ACROSS ALL diffs

  Result: common words (import, const, return) get LOW weight
          rare words (eval, innerHTML, hardcoded_url) get HIGH weight

  The key insight: TF-IDF automatically discovers the same signals
  we hand-crafted in Session 10 — plus ones we didn't think of.
""")

# Fit TF-IDF on the diffs
# max_features=100 keeps the feature vector manageable
# sublinear_tf=True dampens very common terms
tfidf = TfidfVectorizer(
    max_features=100,
    sublinear_tf=True,
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',  # match code identifiers
    ngram_range=(1, 2),                          # single words + pairs
)

features_tfidf = tfidf.fit_transform(diffs).toarray()

print(f"  TF-IDF feature matrix: {features_tfidf.shape}")  # [N, 100]

# Show the top features (most distinctive words across all diffs)
feature_names_tfidf = tfidf.get_feature_names_out()
mean_weights = features_tfidf.mean(axis=0)
top_indices = mean_weights.argsort()[::-1][:20]

print(f"\n  Top 20 most distinctive tokens across all diffs:")
for i, idx in enumerate(top_indices):
    print(f"    {i+1:2d}. {feature_names_tfidf[idx]:30s}  weight={mean_weights[idx]:.4f}")

# Now the interesting part: which TF-IDF features differ between PASS and FAIL?
pass_mask = labels_arr == 0
fail_mask = labels_arr == 1

if pass_mask.sum() > 0 and fail_mask.sum() > 0:
    pass_means = features_tfidf[pass_mask].mean(axis=0)
    fail_means = features_tfidf[fail_mask].mean(axis=0)
    diff_weights = fail_means - pass_means
    top_fail_indices = diff_weights.argsort()[::-1][:10]
    top_pass_indices = diff_weights.argsort()[:10]

    print(f"\n  Tokens most associated with FAIL (higher in FAIL diffs):")
    for idx in top_fail_indices:
        print(f"    {feature_names_tfidf[idx]:30s}  FAIL={fail_means[idx]:.4f}  PASS={pass_means[idx]:.4f}")

    print(f"\n  Tokens most associated with PASS (higher in PASS diffs):")
    for idx in top_pass_indices:
        print(f"    {feature_names_tfidf[idx]:30s}  PASS={pass_means[idx]:.4f}  FAIL={fail_means[idx]:.4f}")

x = 3  # 🔴 BREAKPOINT — Line 176: inspect features_tfidf, feature_names_tfidf
# This is the aha moment:
#   - TF-IDF found tokens YOU didn't think to look for
#   - Some tokens correlate with FAIL that aren't in the 16 regex features
#   - The model will have 100 features instead of 16 — much more to learn from
#
# Compare: Session 10 hand-crafted "eval_calls" vs TF-IDF finding "eval" automatically
# TF-IDF also finds bigrams like "hard coded" or "api key" — more context.


# ============================================================
# PART 4: Stack features — combine hand-crafted + TF-IDF
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Feature stacking — combine all feature types")
print(f"{'='*60}")

print("""
  Feature stacking = concatenate different feature vectors side by side.

  Hand-crafted [16]  — what YOU know matters (eval, secrets, Angular)
  TF-IDF       [100] — what the DATA says matters (automatic)

  Stacked: [16 + 100] = [116] features per diff
""")

# Normalise hand-crafted features (they're raw counts, TF-IDF is already normalised)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_v1_scaled = scaler.fit_transform(features_v1)

# Stack them
features_stacked = np.hstack([features_v1_scaled, features_tfidf])
print(f"  Stacked feature matrix: {features_stacked.shape}")  # [N, 116]

x = 4  # 🔴 BREAKPOINT — Line 209: inspect features_stacked
# Shape is [N, 116] — 16 hand-crafted + 100 TF-IDF
# Why scale? Hand-crafted features range from 0 to ~300 (lines_added).
# TF-IDF features range from 0 to ~1.
# Without scaling, the model would ignore TF-IDF because the numbers are tiny.
# StandardScaler makes every feature mean=0, std=1.


# ============================================================
# PART 5: Train and compare — which features work best?
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Train three models — compare feature types")
print(f"{'='*60}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    input_dim: int,
    hidden: int = 32,
    epochs: int = 300,
    patience: int = 30,
) -> dict:
    """Train a model and return evaluation metrics."""
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Build model — right-sized for the data
    model = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden // 2, 1),
        nn.Sigmoid()
    )

    # Class weights to handle imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    loss_fn = nn.BCELoss(weight=None)  # We'll weight manually if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}]")
    print(f"    Input dim: {input_dim}, Hidden: {hidden}, Params: {param_count}")
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train with early stopping
    best_val_loss = float('inf')
    best_weights = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    # Restore best and evaluate
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_pred_labels = (test_pred > 0.5).float()

    TP = ((test_pred_labels == 1) & (y_test_t == 1)).sum().item()
    TN = ((test_pred_labels == 0) & (y_test_t == 0)).sum().item()
    FP = ((test_pred_labels == 1) & (y_test_t == 0)).sum().item()
    FN = ((test_pred_labels == 0) & (y_test_t == 1)).sum().item()

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Probability spread — how confident is the model?
    probs = test_pred.squeeze().numpy()
    prob_spread = probs.std()

    print(f"    Accuracy:  {accuracy:.1%}")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1:        {f1:.1%}")
    print(f"    Prob spread (std): {prob_spread:.3f}  {'(good — spread out)' if prob_spread > 0.15 else '(bad — clustered near 0.5)'}")
    print(f"    Stopped at epoch: {epoch + 1}")

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prob_spread": prob_spread,
        "probabilities": probs,
        "model": model,
        "param_count": param_count,
    }


# Train three models with different feature sets
print("\nTraining Model A: Hand-crafted only (Session 10 baseline)...")
result_a = train_and_evaluate(
    features_v1_scaled, labels_arr, "Hand-crafted (16)", input_dim=16, hidden=32,
)

print("\nTraining Model B: TF-IDF only (automatic features)...")
result_b = train_and_evaluate(
    features_tfidf, labels_arr, "TF-IDF (100)", input_dim=100, hidden=64,
)

print("\nTraining Model C: Stacked (hand-crafted + TF-IDF)...")
result_c = train_and_evaluate(
    features_stacked, labels_arr, "Stacked (116)", input_dim=116, hidden=64,
)

x = 5  # 🔴 BREAKPOINT — Line 311: inspect result_a, result_b, result_c
# Compare the three models:
#   - Which has highest accuracy? Highest F1?
#   - Which has the best probability spread? (farther from 0.5 = more confident)
#   - result_a.probabilities — are they all clustered near 0.5? That's the problem.
#   - result_c.probabilities — hopefully more spread out.
#
# Try in Evaluate:
#   np.histogram(result_a['probabilities'], bins=10)
#   np.histogram(result_c['probabilities'], bins=10)


# ============================================================
# PART 6: Comparison summary
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Head-to-head comparison")
print(f"{'='*60}")

results = [result_a, result_b, result_c]

print(f"\n  {'Model':<25s} {'Accuracy':>10s} {'F1':>8s} {'Prob Spread':>12s} {'Params':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
for r in results:
    spread_icon = "✓" if r["prob_spread"] > 0.15 else "✗"
    print(f"  {r['name']:<25s} {r['accuracy']:>9.1%} {r['f1']:>7.1%} "
          f"   {r['prob_spread']:.3f} {spread_icon}   {r['param_count']:>7d}")

best = max(results, key=lambda r: r["f1"])
print(f"\n  Best by F1: {best['name']} ({best['f1']:.1%})")

# Probability distribution comparison
print(f"\n  Probability distributions (how confident is each model?):")
for r in results:
    p = r["probabilities"]
    near_half = ((p > 0.4) & (p < 0.6)).sum()
    confident = ((p < 0.2) | (p > 0.8)).sum()
    print(f"    {r['name']:<25s}: {near_half}/{len(p)} uncertain (0.4-0.6), "
          f"{confident}/{len(p)} confident (<0.2 or >0.8)")

x = 6  # 🔴 BREAKPOINT — Line 347: inspect results, probability distributions
# THE key metric is probability spread:
#   Session 10 model: all probabilities near 0.5 (uncertain, defaults to FAIL)
#   With TF-IDF: probabilities should spread towards 0 and 1 (confident)
#
# A model that's 70% accurate but CONFIDENT about its predictions is more useful
# than a model that's 50% accurate and always says "I don't know" (0.5).
#
# This is why the eval suite showed 43% — probabilities were 0.52–0.57.
# The model wasn't "wrong", it was uncertain and the 0.5 threshold caught it.


# ============================================================
# PART 7: Save the best model for production
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Save the improved model")
print(f"{'='*60}")

# Save TF-IDF vectorizer alongside model for production use
import pickle

output_dir = os.path.dirname(__file__)

# Save the best model
best_model = best["model"]
model_path = os.path.join(output_dir, "code_classifier_v2.pt")

torch.save({
    'model_state_dict': best_model.state_dict(),
    'feature_type': best['name'],
    'num_features': best_model[0].in_features,  # First linear layer input dim
    'accuracy': best['accuracy'],
    'precision': best['precision'],
    'recall': best['recall'],
    'f1': best['f1'],
    'prob_spread': best['prob_spread'],
    'training_examples': len(labels_arr),
}, model_path)

# Save TF-IDF vectorizer (needed at inference time)
tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf, f)

# Save scaler (needed for hand-crafted features)
scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n  Model saved to:  {model_path}")
print(f"  TF-IDF saved to: {tfidf_path}")
print(f"  Scaler saved to: {scaler_path}")
print(f"  Model size: {os.path.getsize(model_path) / 1024:.1f} KB")

x = 7  # 🔴 BREAKPOINT — Line 393: saved model + artefacts
# In production (leartech-ai-classifier), you'd need:
#   1. code_classifier_v2.pt — model weights
#   2. tfidf_vectorizer.pkl — to transform new diffs at inference time
#   3. feature_scaler.pkl — to normalise hand-crafted features
#
# The TF-IDF vectorizer is FITTED on the training data — it knows
# which words exist and their IDF weights. A new diff gets transformed
# using this same vocabulary. Any new words not seen during training
# are ignored (which is correct — they have no learned weight).


# ============================================================
# PART 8: What this means for production
# ============================================================

print(f"\n{'='*60}")
print("Session 10.5 Complete!")
print(f"{'='*60}")
print(f"""
Results:
  Model A (hand-crafted 16):  accuracy={result_a['accuracy']:.1%}  f1={result_a['f1']:.1%}  spread={result_a['prob_spread']:.3f}
  Model B (TF-IDF 100):       accuracy={result_b['accuracy']:.1%}  f1={result_b['f1']:.1%}  spread={result_b['prob_spread']:.3f}
  Model C (stacked 116):      accuracy={result_c['accuracy']:.1%}  f1={result_c['f1']:.1%}  spread={result_c['prob_spread']:.3f}

  Best: {best['name']}

Key concepts:
  1. Hand-crafted features plateau when most are zero (eval_calls, innerHTML)
  2. TF-IDF automatically finds distinctive tokens — no hand-crafting needed
  3. Feature stacking combines YOUR domain knowledge with automatic discovery
  4. Probability spread is as important as accuracy — confident wrong > uncertain right
  5. StandardScaler is mandatory when combining features of different scales
  6. The TF-IDF vectorizer is an artefact — must be saved alongside the model

Production path (leartech-ai-classifier):
  1. Add tfidf_features() to app/features.py
  2. Load tfidf_vectorizer.pkl at startup alongside the model
  3. Stack features at inference: [16 hand-crafted, 100 TF-IDF] = [116]
  4. Retrain via Session 10.6 eval harness to prove accuracy ≥80%

Next session: 10.6 — Retrain + Eval Harness
  - Run the improved model against the 7 eval test cases
  - Compare to baseline.json
  - Gate: don't deploy if accuracy < 80%
""")
