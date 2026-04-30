# ML Learning — Environment Setup

## Prerequisites

- Python 3.10+ (`python3 --version`)
- PyCharm Professional (for debugging sessions)
- CLion (for C++ sessions 6b only)
- Git access to `~/leartech/leartech-llm-training-data` (training data)

## Virtual Environment Setup

### First time

```bash
cd ~/leartech/ml-learning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scikit-learn pyyaml
```

### Before each session

```bash
cd ~/leartech/ml-learning
source .venv/bin/activate
```

### Verify

```bash
make check          # full environment check
make check-venv     # Python packages only
```

## PyCharm Setup

### 1. Open the project

Open `~/leartech/ml-learning` as a project in PyCharm (File → Open).

### 2. Set the Python interpreter

PyCharm needs to know about the virtual environment:

1. PyCharm → Settings (⌘,) → Project → Python Interpreter
2. Click the gear icon → Add → Existing Environment
3. Browse to: `~/leartech/ml-learning/.venv/bin/python`
4. Click OK

PyCharm will now use the venv with all ML packages installed.

### 3. Running a session

1. Open any session file, e.g. `sessions/10.5-better-features/better_features.py`
2. Set breakpoints by clicking the gutter next to lines marked `🔴 BREAKPOINT`
3. Right-click the file → Debug (not Run)
4. PyCharm will stop at each breakpoint — inspect variables in the Variables pane

### 4. Using the Evaluate window

While stopped at a breakpoint:
- **Evaluate Expression** (Alt+F8 / ⌥F8): type any Python expression
- Examples:
  ```python
  all_features.shape
  result_a['probabilities'].mean()
  np.histogram(result_c['probabilities'], bins=10)
  ```

### 5. Working directory

PyCharm runs scripts from the project root (`ml-learning/`) by default. If a script uses `Path(__file__).parent`, it resolves relative to the script file — this is correct.

If you get file-not-found errors, check:
- Run/Debug Configuration → Working Directory → should be `~/leartech/ml-learning`

## Command Line

Every session can also be run from the terminal without PyCharm:

### Run a session

```bash
cd ~/leartech/ml-learning
source .venv/bin/activate
python sessions/10.5-better-features/better_features.py
```

### Run all completed sessions in order

```bash
cd ~/leartech/ml-learning
source .venv/bin/activate

# Fundamentals
python sessions/01-tensors/tensors.py
python sessions/02-features/features.py
python sessions/03-single-neuron/single_neuron.py
python sessions/04-neural-network/neural_network.py
python sessions/05-forward-pass/forward_pass.py
python sessions/06-loss-function/loss_function.py
python sessions/07-backpropagation/backpropagation.py
python sessions/08-training-loop/training_loop.py
python sessions/09-overfitting/overfitting.py

# Real classifier + debug loop
python sessions/10-our-classifier/classifier.py
python sessions/10.5-better-features/better_features.py
python sessions/10.6-eval-harness/eval_harness.py
python sessions/10.7-fix-distribution/fix_distribution.py
python sessions/10.8-iterate-to-green/iterate_to_green.py

# Pipeline signals
python sessions/11.5-pipeline-signals/pipeline_signals.py
```

### Run a single session with output only (no breakpoints)

On the command line, breakpoints (`x = 1`) are just variable assignments — the script runs straight through and prints all output. This is useful for:
- Verifying a session runs cleanly after changes
- Checking output without opening PyCharm
- CI validation

### Dependencies per session group

| Sessions | Packages needed |
|---|---|
| 1–10, 10.5–10.8 | `torch`, `numpy`, `scikit-learn` |
| 10.6–10.8, 11.5 | `torch`, `numpy`, `scikit-learn`, `pyyaml` |
| 12 (LoRA) | `torch`, `transformers`, `peft`, `bitsandbytes` (TBD) |

Install all at once:
```bash
pip install torch torchvision numpy matplotlib scikit-learn pyyaml
```

Session 12 dependencies will be documented when that session is built.

## Makefile targets

```bash
make help           # show all targets
make check          # full pre-session check (Python, C++, IDEs, GPU, venv)
make check-venv     # Python packages only
make setup-venv     # create .venv
make install-deps   # install Python packages
make list-sessions  # show session plan
```

## Local Ollama — two different uses

Ollama runs locally for **two separate purposes**. They use different tools and serve different workflows:

### 1. Learning sessions (Session 12 only)

Session 12 (LoRA concepts) downloads GPT-2 from Hugging Face and uses `transformers` + `peft` — Python libraries, not Ollama. Ollama is only relevant for **testing the merged model after LoRA** (Session 12b), when you'd run `ollama create qwen-leartech` and compare base vs fine-tuned locally.

Sessions 1–11.5 don't use Ollama at all — they're Our Classifier, pure PyTorch.

### 2. Testing the AI review pipeline (before pushing changes)

This is the main reason to have local Ollama. When you change `aggregate.py`, `review.py`, prompts, or standards, run the local test suite **before pushing** to catch issues like:
- Ollama scoring 0 on regex patterns (caught too late in this session)
- Aggregate logic bugs (set -e, outlier handling)
- Prompt changes that break JSON parsing

```
make review-ollama    # 14 test cases, Ollama only, no API cost
make review-local     # 14 test cases, Ollama + Claude + DeepSeek APIs
```

These targets wrap `~/leartech/leartech-ai-review-worker/run-local.sh` — same code that runs in the Tekton pipeline, tested on your Mac first.

### Setup

```bash
# Install Ollama (if not already)
brew install ollama

# Start the server
ollama serve

# Pull the model (7B for local, 14B is on the cluster)
ollama pull qwen2.5-coder:7b
```

### Verify

```bash
curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin)['models']]"
```

### Local vs Cluster models

| | Local | Cluster (az/gcp) |
|---|---|---|
| **Model** | `qwen2.5-coder:7b` (4.7GB) | `qwen2.5-coder:14b` (9GB) |
| **Hardware** | CPU (Mac) | L4/T4 GPU |
| **Speed** | ~10-30s per review | ~5-15s per review |
| **Quality** | Good for testing | Production quality |
| **Endpoint** | `http://localhost:11434` | `http://ollama.ai-inference.svc.cluster.local:11434` |

The 7B model is smaller and runs on CPU — fine for testing the pipeline mechanics and catching issues before pushing. The 14B on the cluster is what production uses.

### Running AI review locally

The test harness lives in `~/leartech/leartech-ai-review-worker/`:

```bash
cd ~/leartech/leartech-ai-review-worker

# Full test: local Ollama + Claude API + DeepSeek API
./run-local.sh

# Ollama only (no API costs)
./run-local.sh --ollama-only

# API only (skip Ollama)
./run-local.sh --no-ollama
```

This runs 14 test cases (Go, TypeScript, Terraform, YAML, Helm, Dockerfile, JSON — good and bad examples) through the full pipeline: diff → RAG query → LLM review → aggregate → verdict.

API keys are pulled from GCP Secret Manager automatically (`gcloud secrets versions access`).

### When to test locally

- Before pushing changes to `leartech-dockerfiles/ai-review-worker/app/` (aggregate.py, review.py, etc.)
- Before pushing prompt changes to `leartech-llm-training-data/prompts/`
- When debugging model behaviour (e.g. Ollama scoring 0 on regex patterns)
- Session 12b LoRA: test the fine-tuned model locally before uploading to Ollama on the cluster

### Using local Ollama with ML learning sessions

Session 12 uses `transformers` + `peft` (Hugging Face libraries), not Ollama directly. But you can test a LoRA-trained model on Ollama locally:

```bash
# After Session 12b produces a merged GGUF:
ollama create qwen-leartech -f Modelfile
ollama run qwen-leartech "Review this diff: + eval(userInput)"

# Compare base vs fine-tuned:
ollama run qwen2.5-coder:7b "Review this diff: + eval(userInput)"
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'sklearn'`

```bash
source .venv/bin/activate
pip install scikit-learn
```

### `torch.load` weights_only warning

Some sessions use `weights_only=False` when loading our own models. This is safe — we trained the model ourselves. The warning is about loading untrusted models from the internet.

### PyCharm can't find packages

Check the interpreter is set to `.venv/bin/python` (Settings → Python Interpreter). If you installed packages from the terminal, PyCharm picks them up automatically — try restarting PyCharm if it doesn't.

### `FileNotFoundError` for feedback data

Sessions 10+ need the training data repo:
```bash
ls ~/leartech/leartech-llm-training-data/feedback/
```
If missing, clone it:
```bash
cd ~/leartech
git clone https://github.com/mikelear/leartech-llm-training-data.git
```
