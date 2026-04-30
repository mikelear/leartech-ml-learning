# Session 11.6: Deploy v5 Classifier to Production

## This is not a learning exercise — this is a real deployment

Sessions 10.5–11.5 produced model artefacts in learning session directories.
This session deploys them to the actual `leartech-ai-classifier` service
running on both clusters (GCP + Azure) via the leartech JX3 GitOps pipeline.

**Model affected:** Our Classifier
**Type:** Production deployment
**What changes:** `leartech-ai-classifier` pod serves v5 model with 234 features

## What we're deploying

| Version | Features | Eval accuracy | Where it lives now |
|---|---|---|---|
| v1 (current prod) | 16 regex | 43% (3/7) | `leartech-ai-classifier:0.3.0` |
| **v5 (deploying)** | **234 (28 code + 6 pipeline + 200 char n-grams)** | **100% (7/7)** | `sessions/11.5-pipeline-signals/` |

## How leartech deployments work

This is a JX3 GitOps deployment — understanding the flow is as important as the code change.

### The deployment chain

```
1. You push code to leartech-ai-classifier repo
       │
2. Lighthouse webhook fires → Tekton PipelineRun created
       │
3. PR pipeline runs (6 checks):
       │  ├── pr:            build + test + preview deploy
       │  ├── lint:          ruff fmt + lint + mypy strict
       │  ├── ai-review:    3 LLMs review your change (our own pipeline!)
       │  ├── security-scan: gitleaks + semgrep
       │  ├── image-scan:    grype dependency scan
       │  └── dynamic-scan:  nuclei + nikto + nmap on preview
       │
4. /lgtm + /approve → merge to main
       │
5. Release pipeline:
       │  ├── jx-release-version bumps version (0.3.0 → 0.4.0)
       │  ├── Docker image built + pushed to registry
       │  ├── Helm chart packaged + pushed to chart registry
       │  ├── Image cosign-signed
       │  └── jx promote opens auto-PR on GitOps repo
       │
6. GitOps auto-PR:
       │  ├── jx-build-cluster-gsm (GCP) — bumps helmfile version
       │  └── azure/cluster (Az) — bumps helmfile version
       │
7. Boot job reconciles → new pod deployed with v5 model
```

### What changes in each repo

**`leartech-ai-classifier/` (the service)**

```
app/features.py         ← NEW: extract_features_v3() + extract_all_features()
models/code_classifier.pt  ← REPLACED: v5 model weights (was v1)
models/tfidf_char.pkl      ← NEW: char n-gram vectorizer
models/scaler.pkl          ← NEW: feature scaler
app/model.py            ← UPDATED: load new artefacts, wider input dim
```

**`leartech-llm-training-data/` (eval baseline)**

```
evals/baseline.json     ← UPDATED: reflects v5 results (100%, 7/7)
```

**What does NOT change (yet)**

```
leartech-pipeline-catalog/tasks/ai-review/pullrequest.yaml
  ← classifier is NOT wired in as pre-filter yet (separate step)
  ← pullrequest.yaml still calls 3 LLMs directly
```

### Per-cluster registries

The image publishes to different registries per cluster:

| Cluster | Image registry | Chart registry |
|---|---|---|
| GCP | `us-central1-docker.pkg.dev/product-first/oci/leartech-ai-classifier` | `us-central1-docker.pkg.dev/product-first/oci-charts` |
| Az | `modernburro.azurecr.io/leartech-ai-classifier` | `modernburro.azurecr.io` |

Release pipelines on each cluster resolve via `$DOCKER_REGISTRY` / `$DOCKER_REGISTRY_ORG` env vars — you never hardcode registry URLs.

### Config and secrets

The classifier pod uses:

| Resource | Namespace | What it provides |
|---|---|---|
| `ai-review-cluster-config` ConfigMap | `jx` | `CLUSTER_ID` (gcp/az) |
| No secrets needed | — | Model is baked in, no API keys required |

Unlike the AI review worker (which needs Claude/DeepSeek API keys), the classifier is self-contained. No external API calls. CPU only.

## The changes

### 1. features.py — add v3 extractor + pipeline signal support

The current `features.py` has one function: `extract_features()` returning 16 values.

After this deployment:
- `extract_features()` → kept for backwards compatibility
- `extract_features_v3()` → 28 features (16 original + 6 danger + 6 quality)
- `extract_all_features()` → 234 features (28 code + 6 pipeline + 200 TF-IDF)
  - Accepts optional `pipeline_signals` dict
  - Defaults to neutral `[1, 0, 0, 0, 1, 0]` when not provided
  - Features activate automatically when real infra sends signals

### 2. model.py — load v5 artefacts

Currently loads `code_classifier.pt` only. After:
- Loads `code_classifier.pt` (v5 weights, 234 input dim)
- Loads `tfidf_char.pkl` (char n-gram vectorizer)
- Loads `scaler.pkl` (feature scaler)
- `predict()` calls `extract_all_features()` instead of `extract_features()`

### 3. Model artefacts

| File | Size | Purpose |
|---|---|---|
| `code_classifier.pt` | ~70KB | Neural network weights (v5) |
| `tfidf_char.pkl` | ~50KB | Char n-gram vocabulary + IDF weights |
| `scaler.pkl` | ~5KB | StandardScaler fitted on training data |

### 4. Baseline update

`evals/baseline.json` updated to reflect v5 predictions. The eval pipeline on both clusters will compare future models against this new baseline.

## Verification

After deployment, verify on both clusters:

```bash
# GCP
kubectl --context gke_product-first_us-east1-b_tf-jx-usable-bird \
  -n jx-staging exec deploy/leartech-ai-classifier -- \
  curl -s localhost:8080/health | jq .

# Should show:
#   "model_version": "v5"
#   "features": 234
#   "eval_accuracy": 1.0

# Test prediction
kubectl --context gke_product-first_us-east1-b_tf-jx-usable-bird \
  -n jx-staging exec deploy/leartech-ai-classifier -- \
  curl -s -X POST localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"diff": "+ const API_KEY = \"sk-secret\";\n+ eval(input);"}' | jq .

# Should show:
#   "verdict": "FAIL"
#   "confidence": >0.9
```

## What's next after this deploys

| Step | What | Why |
|---|---|---|
| Wire into pipeline | Add classifier call to `pullrequest.yaml` as pre-filter | Skip expensive LLM calls when classifier is confident-FAIL |
| Training CronJob | Weekly retrain on accumulated feedback | Model improves as data grows |
| Session 12 | LoRA concepts (different model — Ollama/Qwen) | Teach the LLM leartech conventions |
