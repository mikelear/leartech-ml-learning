"""
Session 10.6: Eval Harness — Unit Testing for Models

Just like code has tests that gate deployment, models need an eval harness
that gates retraining. This session builds one using the REAL test cases
from leartech-llm-training-data/evals/ and compares the Session 10.5 model
against the Session 10 baseline.

Think of it exactly like unit tests:
  - Fixed inputs with known expected outputs  (test cases)
  - Assertions that must hold                 (accuracy >= 80%)
  - Regression detection                      (nothing that was passing now fails)
  - Runs in CI on every change                (evals/run_evals.py on both clusters)

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.

In production: leartech-llm-training-data/evals/run_evals.py does this on every PR.
"""

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


# ============================================================
# PART 1: Load the test suite (like loading test fixtures)
# ============================================================

print("=" * 60)
print("PART 1: Load the eval test suite")
print("=" * 60)

EVALS_DIR = Path(os.path.expanduser("~/leartech/leartech-llm-training-data/evals"))
MANIFEST = EVALS_DIR / "manifest.yaml"
BASELINE_FILE = EVALS_DIR / "baseline.json"

with open(MANIFEST) as f:
    manifest = yaml.safe_load(f)

test_cases = manifest["test_cases"]

print(f"\n  Loaded {len(test_cases)} test cases from manifest.yaml:")
for tc in test_cases:
    flags = ""
    if tc.get("must_flag"):
        flags = f"  must_flag: {tc['must_flag']}"
    if tc.get("must_not_flag"):
        flags = f"  must_not_flag: {tc['must_not_flag']}"
    print(f"    {tc['verdict']:4s}  {tc['file']}{flags}")

# Load the diffs
print(f"\n  Loading diff files...")
for tc in test_cases:
    diff_path = EVALS_DIR / tc["file"]
    with open(diff_path) as f:
        tc["diff_text"] = f.read()
    print(f"    {tc['file']:50s}  {len(tc['diff_text']):>5d} chars")

x = 1  # 🔴 BREAKPOINT — Line 56: inspect test_cases, manifest
# This is exactly like test fixtures in pytest:
#   - Each test case has an input (diff) and expected output (verdict)
#   - Some have "must_flag" (like asserting specific error messages)
#   - The manifest IS the test suite definition


# ============================================================
# PART 2: Load the baseline (like "last known good" test results)
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Load baseline — the current 'last known good'")
print(f"{'='*60}")

with open(BASELINE_FILE) as f:
    baseline = json.load(f)

print(f"\n  Baseline: {len(baseline)} results")
print(f"\n  {'File':50s} {'Expected':>8s} {'Actual':>8s} {'Prob':>6s} {'Status':>6s}")
print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

baseline_pass = 0
baseline_fail = 0
for b in baseline:
    icon = "✓" if b["match"] else "✗"
    print(f"  {b['file']:50s} {b['expected_verdict']:>8s} {b['actual_verdict']:>8s} "
          f"{b['probability']:.3f} {icon}")
    if b["match"]:
        baseline_pass += 1
    else:
        baseline_fail += 1

baseline_accuracy = baseline_pass / len(baseline)
print(f"\n  Baseline accuracy: {baseline_accuracy:.0%} ({baseline_pass}/{len(baseline)})")
print(f"  Baseline problems: all probabilities cluster at 0.52–0.62")
print(f"  The model can't tell PASS from FAIL — it's guessing.")

x = 2  # 🔴 BREAKPOINT — Line 89: inspect baseline, baseline_accuracy
# This IS the current production model's performance.
# 43% accuracy, 3/7 correct (all 3 FAIL cases pass, all 4 PASS cases fail).
# Probabilities: 0.527–0.619 — clustered right around 0.5.
#
# Compare to unit tests: this is like a test suite where 4/7 tests fail.
# You wouldn't deploy code with 4 failing tests. Same logic applies here.


# ============================================================
# PART 3: Load the Session 10.5 model (the "new code" under test)
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Load the improved model from Session 10.5")
print(f"{'='*60}")

SESSION_105_DIR = Path(__file__).parent.parent / "10.5-better-features"

# Load model
model_path = SESSION_105_DIR / "code_classifier_v2.pt"
checkpoint = torch.load(model_path, weights_only=False)  # Our own model — trusted

input_dim = checkpoint["num_features"]

model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load TF-IDF vectorizer and scaler
with open(SESSION_105_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(SESSION_105_DIR / "feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"\n  Model loaded: {input_dim} input features")
print(f"  Session 10.5 training accuracy: {checkpoint['accuracy']:.1%}")
print(f"  Session 10.5 F1: {checkpoint['f1']:.1%}")


def extract_features_v1(diff: str) -> list[float]:
    """Session 10 hand-crafted features."""
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


def predict(diff: str) -> dict:
    """Run the Session 10.5 model on a diff."""
    # Hand-crafted features (scaled)
    v1_raw = np.array([extract_features_v1(diff)])
    v1_scaled = scaler.transform(v1_raw)

    # TF-IDF features
    tfidf_feat = tfidf.transform([diff]).toarray()

    # Stack
    stacked = np.hstack([v1_scaled, tfidf_feat])
    tensor = torch.tensor(stacked, dtype=torch.float32)

    with torch.no_grad():
        prob = model(tensor).item()

    verdict = "FAIL" if prob > 0.5 else "PASS"
    return {"verdict": verdict, "probability": prob}


print(f"\n  Quick sanity check:")
sanity_pass = predict("+ func handleRequest(w http.ResponseWriter) error {\n+   return nil\n+ }")
sanity_fail = predict("+ const API_KEY = 'sk-secret-key-12345';\n+ eval(userInput);")
print(f"    Clean Go code:     {sanity_pass['verdict']} (prob={sanity_pass['probability']:.3f})")
print(f"    Secrets + eval:    {sanity_fail['verdict']} (prob={sanity_fail['probability']:.3f})")

x = 3  # 🔴 BREAKPOINT — Line 164: inspect model, predict results
# sanity_pass should be low probability (clean code)
# sanity_fail should be high probability (dangerous code)
# If both cluster near 0.5, the model isn't useful.
# If they're far apart (e.g. 0.1 vs 0.9), the model is confident.


# ============================================================
# PART 4: Run the eval suite (like running pytest)
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Run eval suite — the model's 'test run'")
print(f"{'='*60}")

results = []
for tc in test_cases:
    prediction = predict(tc["diff_text"])
    match = prediction["verdict"] == tc["verdict"]
    results.append({
        "file": tc["file"],
        "expected_verdict": tc["verdict"],
        "actual_verdict": prediction["verdict"],
        "probability": round(prediction["probability"], 4),
        "match": match,
        "status": "pass" if match else "fail",
    })

print(f"\n  {'File':50s} {'Expected':>8s} {'Actual':>8s} {'Prob':>6s} {'Status':>6s}")
print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

new_pass = 0
new_fail = 0
for r in results:
    icon = "✓" if r["match"] else "✗"
    print(f"  {r['file']:50s} {r['expected_verdict']:>8s} {r['actual_verdict']:>8s} "
          f"{r['probability']:.3f}  {icon}")
    if r["match"]:
        new_pass += 1
    else:
        new_fail += 1

new_accuracy = new_pass / len(results)
print(f"\n  New model accuracy: {new_accuracy:.0%} ({new_pass}/{len(results)})")

x = 4  # 🔴 BREAKPOINT — Line 203: inspect results
# Compare each result to the baseline.
# Are the probabilities more spread out? (good)
# Did any PASS case that was wrong in baseline become right? (improvement)
# Did any case that was right become wrong? (regression — bad!)


# ============================================================
# PART 5: Compare to baseline (regression detection)
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Regression detection — compare new vs baseline")
print(f"{'='*60}")

print("""
  Regression = something that was working now breaks.

  In code: a test that was passing now fails after your change.
  In ML:   a test case the old model got right, the new model gets wrong.

  A model that's 90% accurate but regressed on a security test case
  is WORSE than a model that's 80% accurate with no regressions.
  Regressions mean you broke something that was working.
""")

baseline_map = {b["file"]: b for b in baseline}
improved = []
regressed = []
unchanged = []

for r in results:
    b = baseline_map.get(r["file"])
    if b is None:
        improved.append({"file": r["file"], "reason": "new test case"})
        continue
    if r["match"] and not b["match"]:
        improved.append({
            "file": r["file"],
            "reason": f"was {b['actual_verdict']}(prob={b['probability']:.3f}), "
                      f"now {r['actual_verdict']}(prob={r['probability']:.3f})"
        })
    elif not r["match"] and b["match"]:
        regressed.append({
            "file": r["file"],
            "reason": f"was {b['actual_verdict']}(prob={b['probability']:.3f}), "
                      f"now {r['actual_verdict']}(prob={r['probability']:.3f})"
        })
    else:
        unchanged.append(r["file"])

print(f"\n  Improved:  {len(improved)}")
for i in improved:
    print(f"    ↑ {i['file']}")
    print(f"      {i['reason']}")

print(f"\n  Regressed: {len(regressed)}")
for r in regressed:
    print(f"    ↓ {r['file']}")
    print(f"      {r['reason']}")

print(f"\n  Unchanged: {len(unchanged)}")
for u in unchanged:
    print(f"    = {u}")

x = 5  # 🔴 BREAKPOINT — Line 250: inspect improved, regressed, unchanged
# Regressions are THE critical check.
# If regressed is not empty, you should NOT deploy the new model.
# Even if accuracy went up overall, a regression means you broke something.
#
# This is the exact same logic as: "don't merge if any test fails,
# even if you added 10 new passing tests."


# ============================================================
# PART 6: Gate decision (like CI pass/fail)
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Gate decision — should we deploy this model?")
print(f"{'='*60}")

ACCURACY_FLOOR = 0.80  # Don't deploy below 80%
ALLOW_REGRESSIONS = False  # Regressions block deployment

gate_passed = True
gate_reasons = []

# Check 1: accuracy floor
if new_accuracy < ACCURACY_FLOOR:
    gate_passed = False
    gate_reasons.append(f"Accuracy {new_accuracy:.0%} < floor {ACCURACY_FLOOR:.0%}")
else:
    gate_reasons.append(f"✓ Accuracy {new_accuracy:.0%} >= floor {ACCURACY_FLOOR:.0%}")

# Check 2: no regressions
if regressed and not ALLOW_REGRESSIONS:
    gate_passed = False
    gate_reasons.append(f"✗ {len(regressed)} regression(s) detected")
else:
    gate_reasons.append(f"✓ No regressions")

# Check 3: improvement over baseline
if new_accuracy > baseline_accuracy:
    gate_reasons.append(f"✓ Accuracy improved: {baseline_accuracy:.0%} → {new_accuracy:.0%}")
elif new_accuracy == baseline_accuracy:
    gate_reasons.append(f"= Accuracy unchanged: {new_accuracy:.0%}")
else:
    gate_reasons.append(f"↓ Accuracy dropped: {baseline_accuracy:.0%} → {new_accuracy:.0%}")

# Check 4: probability spread (confidence)
baseline_probs = [b["probability"] for b in baseline]
new_probs = [r["probability"] for r in results]
baseline_spread = np.std(baseline_probs)
new_spread = np.std(new_probs)

if new_spread > baseline_spread:
    gate_reasons.append(f"✓ Probability spread improved: {baseline_spread:.3f} → {new_spread:.3f}")
else:
    gate_reasons.append(f"↓ Probability spread decreased: {baseline_spread:.3f} → {new_spread:.3f}")

print(f"\n  Gate: {'PASS ✅' if gate_passed else 'FAIL ❌'}")
print(f"\n  Checks:")
for reason in gate_reasons:
    print(f"    {reason}")

print(f"\n  Summary:")
print(f"    Baseline:  {baseline_accuracy:.0%} accuracy, {baseline_spread:.3f} spread")
print(f"    New model: {new_accuracy:.0%} accuracy, {new_spread:.3f} spread")
print(f"    Improved:  {len(improved)} cases")
print(f"    Regressed: {len(regressed)} cases")

x = 6  # 🔴 BREAKPOINT — Line 310: inspect gate_passed, gate_reasons
# This is the CI gate for models:
#   accuracy >= 80%     (like coverage >= 80%)
#   no regressions      (like no test failures)
#   spread improved     (like no new warnings)
#
# In production, evals/run_evals.py does exactly this on every PR
# to leartech-llm-training-data. It posts the results as a PR comment.


# ============================================================
# PART 7: Save new baseline (like updating snapshots)
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Save new baseline")
print(f"{'='*60}")

if gate_passed:
    new_baseline_path = os.path.join(os.path.dirname(__file__), "new_baseline.json")
    with open(new_baseline_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  New baseline saved to: {new_baseline_path}")
    print(f"  To update production baseline:")
    print(f"    cp {new_baseline_path} {BASELINE_FILE}")
    print(f"\n  Like updating test snapshots: only when the new output is correct.")
else:
    print(f"\n  Gate FAILED — not saving new baseline.")
    print(f"  Fix the model first, then re-run.")
    print(f"\n  Like a failing test: don't update the snapshot if the code is wrong.")


# ============================================================
# PART 8: Summary
# ============================================================

print(f"\n{'='*60}")
print("Session 10.6 Complete!")
print(f"{'='*60}")
print(f"""
The eval harness = unit tests for models:

  | Code testing          | Model eval                          |
  |-----------------------|-------------------------------------|
  | test_login.py         | test_cases/fail/hardcoded-secrets   |
  | assert status == 200  | assert verdict == "FAIL"            |
  | coverage >= 80%       | accuracy >= 80%                     |
  | no test failures      | no regressions                      |
  | update snapshots      | update baseline.json                |
  | CI blocks merge       | pipeline blocks deploy              |

Results:
  Baseline (Session 10): {baseline_accuracy:.0%} accuracy, {baseline_spread:.3f} spread
  New (Session 10.5):    {new_accuracy:.0%} accuracy, {new_spread:.3f} spread
  Gate: {'PASS ✅' if gate_passed else 'FAIL ❌'}

Production path:
  1. leartech-llm-training-data/evals/run_evals.py — runs on every PR
  2. Posts results as PR comment: "Standards Eval Results [az/gcp]"
  3. Compares to evals/baseline.json — blocks on regressions
  4. Currently does NOT block (exit code always 0) — Session 10.6
     shows why it should block on regressions + accuracy floor

Next session: 11.5 — Pipeline Signals as Features
  - Add risk-assessor, e2e, Tempo traces as classifier features
  - Show how infrastructure data improves code quality predictions
""")
