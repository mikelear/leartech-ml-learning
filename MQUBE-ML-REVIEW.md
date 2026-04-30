# Mqube ML Estate — Review for Leartech

_Authored: 2026-04-30. Audience: Leartech engineers (specifically anyone working in `~/leartech/ml-learning` or building Leartech's first internal ML services)._

This document inventories the **Mqube** ML estate as a reference for **Leartech**. Mqube and Leartech are separate orgs with separate GitOps; nothing here is currently deployed by Leartech. The point of the write-up is so Leartech can borrow the patterns Mqube has already battle-tested, rather than rediscover them.

## Scope and method

- **Source repos surveyed**: all 468 directories under `~/mqubeRepos/`. Filtered to those that contain real ML code — defined as "trains, fine-tunes, or runs inference on a model whose weights live with the codebase or are loaded from a model store" — explicitly excluding repos whose only "AI" surface is calling a hosted LLM API (Azure OpenAI, Claude, OpenAI client, etc.).
- **Production ground truth**: the Mqube prod GitOps repo `JX3_Azure_Vault_Production` (https://github.com/spring-financial-group/JX3_Azure_Vault_Production), namespace helmfile at `helmfiles/jx-jx3-azure-vault-production/helmfile.yaml`. Last commit observed: 2026-04-29 15:03 (auto-promote merge `#6326`) — repo is alive and auto-promoting daily.
- **Note on the org boundary**: the GitHub org is `spring-financial-group`, not `mqube`. Mqube was historically rebranded; the older `environment-mqube-production` and `environment-ninjabald-production` repos in `~/mqubeRepos/` are JX2-era artifacts last touched 2020–2021 and are **not** the live prod. Don't be misled by the names.

## What Mqube actually runs in production

Eleven services in the prod namespace are ML-adjacent. Not all of them are real ML — Mqube's naming convention puts `mqube-ml-` on the front of anything that participates in the document/transaction processing pipeline, including pure heuristics. Reading the names is not enough; reading the imports is.

### Real ML services (8)

Versions are the helm chart version pinned in the prod helmfile.

| Service | Version | Framework | Task |
|---|---|---|---|
| `mqube-transaction-categorisation-service` | 2.0.0 | PyTorch + transformers | Fine-tuned BERT, multi-class transaction category prediction |
| `mqube-ml-bs-summary-ner-service` | 2.0.0 | PyTorch + transformers | BERT-family token classification on bank-statement summaries |
| `mqube-ml-doc-classifier-service` | 2.0.1 | PyTorch + transformers | Multi-label document classification (fine-tuned transformer) |
| `mqube-ml-ner-bs-table-finder-service` | 3.0.0 | YOLOv8 (Ultralytics) | Computer vision — detects table regions on bank-statement page images |
| `mqube-case-complexity-service` | 1.0.0 | XGBoost + scikit-learn | Tabular case-complexity scoring |
| `mqube-ml-ner-service` | 1.0.1 | OpenAI + Ollama + OpenCV | Hybrid — partly heuristic, partly hosted LLM. Marginal. |
| `mqube-ml-google-ocr-service` | 2.0.2 | Google Cloud Vision API | Hosted OCR (no local model). Included for completeness. |
| `mqube-ml-doc-classifier-service` config | — | — | This is the only ML release in the prod helmfile with a per-service `configs/*.yaml` override, suggesting bespoke prod wiring. |

### Named `ml-` but actually heuristic (3)

These are deployed in prod and named like ML services, but contain no ML framework dependencies. Useful as deployment-pattern references; not relevant as ML references.

| Service | Version | Reality |
|---|---|---|
| `mqube-ml-ner-bs-transactions-extraction-service` | 2.0.1 | Pure geometric reasoning over OCR words (numpy + pandas + OpenCV) to parse tables into transaction rows. Despite the name, no `torch` / `transformers` / `tensorflow`. |
| `mqube-ml-doc-joiner-service` | 1.0.1 | `textdistance` heuristics for stitching pages together. |
| `mqube-ml-doc-splitter-service` | 2.0.0 | Rule-based page boundary detection. |

### Cross-cutting / utility (2)

| Service | Version | Role |
|---|---|---|
| `mqube-ml-evaluation-service` | 1.1.0 | FastAPI metrics service. Consumes predictions + ground truth, returns precision/recall/CER/WER (`jiwer`, `textdistance`). MongoDB-backed for ground-truth persistence + Azure Blob for artifacts. **No model in this service** — it scores other services' outputs. |
| `mqube-ml-model-manager` | not in prod helmfile | CLI / library used by training pipelines to push artifacts to Azure Blob. Consumed by services at startup to fetch the right model. Pipeline-side, not deployment-side. |

## The architectural pattern

Mqube has converged on a single pattern for every ML capability:

1. **Two repos per capability**: a `…-service` repo (FastAPI inference) and a `…-train` repo (PyTorch Lightning training). Examples: `mqube-transaction-categorisation-service` ↔ `mqube-transaction-categorisation-train`, `mqube-ml-doc-classifier-service` ↔ `mqube-ml-doc-classifier-train`, `mqube-ml-bs-summary-ner-service` ↔ `mqube-ml-bs-summary-ner-train`.
2. **Decoupled artifact handoff** via `mqube-ml-model-manager` — training pushes a versioned `.pt` (or equivalent) to Azure Blob; the service fetches it at startup. Neither side knows about the other's git history. The model's "version" is independent of the chart version.
3. **Train repos are not deployed in the prod namespace**. None of the `…-train` repos appear in `helmfiles/jx-jx3-azure-vault-production/helmfile.yaml`. Training runs on a build/training cluster (Tekton tasks or a separate JX env); only the artifact crosses into prod via Blob.
4. **A central scaffold**: `mqube-ml-train-template` is the PyTorch Lightning + MongoDB-ingest + Blob-output skeleton that new training repos copy from. There's no equivalent service-side template; service repos appear hand-rolled (and slightly inconsistent — `doc-classifier-service` is the only one with a per-service config override in prod).
5. **Out-of-the-loop evaluation**: `mqube-ml-evaluation-service` is offline-style. It takes predictions and ground truth as input — it doesn't intercept live production traffic. Pluggable evaluators per document type (Bank Statements, Payslips, Tax forms).

## What's not there

These are absent from Mqube's ML estate, not just from prod:

- **No sentence-transformers, no spaCy, no gensim** — no separate embeddings path. All semantic work is delegated to fine-tuned BERT heads.
- **No vector store / retrieval / RAG repos** — if RAG happens, it lives inside an LLM-API service like `mqube-underwriting-agent-service`, which is pure OpenAI-orchestration and not part of this review.
- **No active LightGBM. XGBoost appears once** (case-complexity) and only on the inference side — the fitted model is produced outside the visible repos, possibly a notebook.
- **No PEFT / LoRA / adapter-based fine-tuning** of any owned model. Every `…-train` repo does **full fine-tuning** of small BERT bases. Relevant context: Session 12 of the ml-learning journey (LoRA concepts) has no internal precedent to copy.
- **No forecasting, anomaly-detection, fraud-scoring** repos. XGBoost on case-complexity is the only tabular-ML repo.
- **No online evaluation / shadow-mode / A-B harness** — `mqube-ml-evaluation-service` is offline-only.

## Dead-on-arrival repos (in `~/mqubeRepos/`, deployed nowhere)

A 2020-era data-science effort left behind: `mqube-ds-image-processing` (TF 2.2 + Keras), `mqube-ds-face-detection` (facenet-pytorch), `mqube-ds-rec-template` (mixed PyTorch/TF), `mqube-ds-craft`, `mqube-craft-api`, `mqube-starnet-api`. None appear in any prod, dev, or preview helmfile. Treat as historical; do not use as references.

## Implications for Leartech

Leartech currently has **zero internal ML services in production**. `leartech-ai-classifier` is the only Leartech repo serving an `/openapi.json` from a FastAPI app, and it calls a hosted LLM — no local model. Anything Leartech builds is greenfield.

The Mqube patterns worth copying directly into Leartech:

- **Two-repo split (`…-service` + `…-train`)** — strongly recommended. Releases decouple, training runs on a different schedule from serving, and the model artifact becomes a clean handoff. Mqube has run this for years; the pattern works.
- **A model-manager equivalent** — Leartech doesn't have one. Before the second ML service ships, build `leartech-ml-model-manager` (or pick a name) that pushes/pulls model artifacts to Azure Blob with versioning. Don't let each service hand-roll its own model fetch.
- **PyTorch Lightning for training** — Mqube standardised on this. No reason for Leartech to differ unless there's a specific need.
- **Offline evaluation as its own service** — `mqube-ml-evaluation-service`'s shape (FastAPI, pluggable evaluators, MongoDB for ground-truth) is a reasonable template once Leartech has more than one ML service to compare. Start in-memory, add persistence when needed.
- **GPU-only where required** — Mqube's transaction-categorisation, doc-classifier, and bs-summary-ner have GPU node selectors; the heuristic services run on CPU with HPA 2–8 and 500% CPU thresholds. Right-size per service rather than blanket GPU.

The Mqube patterns to **not** copy:

- **Naming that lies** — Mqube's `mqube-ml-` prefix is overloaded onto pure-heuristic services (`doc-joiner`, `doc-splitter`, `ner-bs-transactions-extraction`). New readers can't tell which services have models from names alone. Leartech should reserve `ml-` for repos with actual ML.
- **Inconsistent service scaffolds** — only one of Mqube's ML services has a `configs/*.yaml` in the prod helmfile, suggesting service-side patterns drifted. Leartech should ship a service-side template alongside any train-side template, or copy one cleanly.
- **No online evaluation** — Mqube has no shadow-mode or A-B harness. If Leartech is building from scratch, it's cheaper to wire prediction-logging in from day one than to retrofit it later.

Gaps Mqube has not filled, that Leartech could lead on:

- **PEFT / LoRA fine-tuning** of an owned base model. Session 12 of the ml-learning journey is the natural starting point; there is no Mqube precedent to follow.
- **Vector store / retrieval** as a first-class service — `chromadb` is already running in `ai-inference` on the `modern-burro` Azure cluster but no Leartech or Mqube service consumes it as a managed dependency.
- **Online evaluation harness** — see above.

## Reading list for Session 11 (deploy `code_classifier.pt` to Azure)

The user's Session 11 builds a Flask/FastAPI service around the `~5KB` `code_classifier.pt` from Session 10 and deploys it to the `modern-burro` Azure cluster. Pre-reading from Mqube, in order:

1. **`mqube-ml-ner-bs-transactions-extraction-service`** (heuristic baseline). The skeleton: FastAPI bootstrap (`app/main.py:1–18`), helm chart at `charts/mqube-ml-ner-bs-transactions-extraction-service/templates/deployment.yaml`, HPA 2–8 with 500% CPU threshold, 60s liveness / 10s readiness in `values.yaml`. Read this first to see what a Mqube ML chart looks like with the ML stripped out. Skip the geometric OCR logic in `app/services/`.
2. **`mqube-transaction-categorisation-service`** (closest analogue). PyTorch + BERT + FastAPI with model fetch from `mqube-ml-model-manager` → Azure Blob, GPU node selector. Read the model loader (how a `.pt` is pulled and pinned at startup), the FastAPI route, and the chart's `deployment.yaml` for resources + node selector. Skip the BERT tokenizer specifics — the Session 10 classifier doesn't tokenize.
3. **`mqube-ml-evaluation-service`** (post-deploy). FastAPI for precision/recall/edit-distance over the wire — direct continuation of Session 10's confusion-matrix work and the natural Session 11 follow-up. Read `app/controllers/evaluation.py:18–36` (endpoint shape), `app/services/evaluation/service.py:34–68` (per-document-type evaluator dispatch). Skip the MongoDB persistence and Azure Blob ground-truth storage; start in-memory.

Read order matters: skeleton first, then add the ML, then add the metrics on top. That's the same path the Session 11 build can follow.
