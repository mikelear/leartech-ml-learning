# ML Learning Journey — Pre-Session Checks & Setup
#
# Run 'make' to see all available targets
# Run 'make check' before each session to verify your environment

PYTHON := python3
PIP := pip3
VENV_DIR := .venv
CLUSTER_CONTEXT := modern-burro

# Minimum versions
MIN_PYTHON_MAJOR := 3
MIN_PYTHON_MINOR := 10
MIN_CMAKE_MAJOR := 3
MIN_CMAKE_MINOR := 18

# Colours
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m

help:
	@echo ""
	@echo "  ML Learning Journey"
	@echo "  ==================="
	@echo ""
	@echo "  Setup (see SETUP.md for full guide):"
	@echo "    check            Run all pre-session checks"
	@echo "    check-python     Check Python and pip"
	@echo "    check-cpp        Check C++ toolchain (clang, cmake)"
	@echo "    check-ide        Check PyCharm and CLion"
	@echo "    check-gpu        Check Azure GPU node and Ollama"
	@echo "    check-ollama     Check local Ollama server and models"
	@echo "    check-venv       Check virtual environment and packages"
	@echo ""
	@echo "    setup-venv       Create Python virtual environment"
	@echo "    install-deps     Install Python dependencies (PyTorch, numpy, etc.)"
	@echo "    setup-cpp        Download libtorch for C++ sessions"
	@echo ""
	@echo "  Sessions:"
	@echo "    list-sessions    Show all sessions and progress"
	@echo "    run SESSION=10.5 Run a session from the command line"
	@echo ""
	@echo "  Local AI Review Testing:"
	@echo "    review-local     Run AI review test suite (Ollama + APIs)"
	@echo "    review-ollama    Run AI review test suite (Ollama only, no API cost)"
	@echo "    review-api       Run AI review test suite (Claude + DeepSeek only)"
	@echo ""
	@echo "  Run 'make check' before your first session."
	@echo "  See SETUP.md for PyCharm interpreter setup and troubleshooting."
	@echo ""

# ===========================
# Full pre-session check
# ===========================

check: _header check-python check-cpp check-ide check-gpu check-venv _footer

_header:
	@echo ""
	@echo "============================================"
	@echo "  ML Learning Journey — Pre-Session Check"
	@echo "============================================"
	@echo ""

_footer:
	@echo ""
	@echo "============================================"
	@echo "  Check complete"
	@echo "============================================"
	@echo ""

# ===========================
# Python checks
# ===========================

check-python:
	@echo "--- Python ---"
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		VERSION=$$($(PYTHON) --version 2>&1 | awk '{print $$2}'); \
		MAJOR=$$(echo $$VERSION | cut -d. -f1); \
		MINOR=$$(echo $$VERSION | cut -d. -f2); \
		if [ "$$MAJOR" -ge $(MIN_PYTHON_MAJOR) ] && [ "$$MINOR" -ge $(MIN_PYTHON_MINOR) ]; then \
			echo "  $(GREEN)✓$(NC) Python $$VERSION"; \
		else \
			echo "  $(RED)✗$(NC) Python $$VERSION (need $(MIN_PYTHON_MAJOR).$(MIN_PYTHON_MINOR)+)"; \
		fi; \
	else \
		echo "  $(RED)✗$(NC) Python not found — install from python.org"; \
	fi
	@if command -v $(PIP) >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) pip $$($(PIP) --version 2>&1 | awk '{print $$2}')"; \
	else \
		echo "  $(RED)✗$(NC) pip not found"; \
	fi
	@echo ""

# ===========================
# C++ checks
# ===========================

check-cpp:
	@echo "--- C++ Toolchain ---"
	@if command -v clang++ >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) clang++ $$(clang++ --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"; \
	elif command -v g++ >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) g++ $$(g++ --version 2>&1 | head -1)"; \
	else \
		echo "  $(RED)✗$(NC) No C++ compiler found — install Xcode command line tools: xcode-select --install"; \
	fi
	@if command -v cmake >/dev/null 2>&1; then \
		VERSION=$$(cmake --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1); \
		echo "  $(GREEN)✓$(NC) CMake $$VERSION"; \
	else \
		echo "  $(RED)✗$(NC) CMake not found — brew install cmake"; \
	fi
	@if command -v make >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) Make available"; \
	else \
		echo "  $(RED)✗$(NC) Make not found"; \
	fi
	@echo ""

# ===========================
# IDE checks
# ===========================

check-ide:
	@echo "--- IDEs ---"
	@if [ -d "/Applications/PyCharm Professional.app" ] || \
		[ -d "/Applications/PyCharm.app" ] || \
		mdfind "kMDItemFSName == 'PyCharm Professional.app'" 2>/dev/null | grep -q PyCharm; then \
		echo "  $(GREEN)✓$(NC) PyCharm installed"; \
	else \
		echo "  $(RED)✗$(NC) PyCharm not found — install via JetBrains Toolbox"; \
	fi
	@if [ -d "/Applications/CLion.app" ] || \
		mdfind "kMDItemFSName == 'CLion.app'" 2>/dev/null | grep -q CLion; then \
		echo "  $(GREEN)✓$(NC) CLion installed"; \
	else \
		echo "  $(YELLOW)⚠$(NC) CLion not found — needed for C++ sessions (install via JetBrains Toolbox)"; \
	fi
	@echo "  $(YELLOW)ℹ$(NC) Check Claude plugin in both IDEs:"
	@echo "    PyCharm → Settings → Plugins → search 'Claude'"
	@echo "    CLion   → Settings → Plugins → search 'Claude'"
	@echo ""

# ===========================
# GPU / Cluster checks
# ===========================

check-gpu:
	@echo "--- Azure GPU Cluster ---"
	@if command -v kubectl >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) kubectl available"; \
	else \
		echo "  $(RED)✗$(NC) kubectl not found"; \
		exit 0; \
	fi
	@if kubectl --context $(CLUSTER_CONTEXT) cluster-info >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) Azure cluster reachable"; \
	else \
		echo "  $(RED)✗$(NC) Azure cluster not reachable (context: $(CLUSTER_CONTEXT))"; \
		exit 0; \
	fi
	@GPU_NODE=$$(kubectl --context $(CLUSTER_CONTEXT) get nodes -l node.kubernetes.io/instance-type=Standard_NC4as_T4_v3 --no-headers 2>/dev/null | awk '{print $$1}'); \
	if [ -n "$$GPU_NODE" ]; then \
		echo "  $(GREEN)✓$(NC) GPU node: $$GPU_NODE (T4 16GB)"; \
	else \
		echo "  $(RED)✗$(NC) No GPU node found"; \
	fi
	@OLLAMA=$$(kubectl --context $(CLUSTER_CONTEXT) get pods -n ai-inference --no-headers 2>/dev/null | grep ollama | grep Running | awk '{print $$1}'); \
	if [ -n "$$OLLAMA" ]; then \
		echo "  $(GREEN)✓$(NC) Ollama running: $$OLLAMA"; \
	else \
		echo "  $(YELLOW)⚠$(NC) Ollama not running"; \
	fi
	@CHROMA=$$(kubectl --context $(CLUSTER_CONTEXT) get pods -n ai-inference --no-headers 2>/dev/null | grep chromadb | grep Running | awk '{print $$1}'); \
	if [ -n "$$CHROMA" ]; then \
		echo "  $(GREEN)✓$(NC) ChromaDB running: $$CHROMA"; \
	else \
		echo "  $(YELLOW)⚠$(NC) ChromaDB not running"; \
	fi
	@echo ""

# ===========================
# Virtual environment checks
# ===========================

check-venv:
	@echo "--- Python Virtual Environment ---"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "  $(GREEN)✓$(NC) Virtual environment exists at $(VENV_DIR)"; \
		if [ -f "$(VENV_DIR)/bin/python" ]; then \
			TORCH=$$($(VENV_DIR)/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null); \
			if [ -n "$$TORCH" ]; then \
				echo "  $(GREEN)✓$(NC) PyTorch $$TORCH"; \
				CUDA=$$($(VENV_DIR)/bin/python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU only')" 2>/dev/null); \
				echo "  $(GREEN)✓$(NC) Device: $$CUDA"; \
			else \
				echo "  $(YELLOW)⚠$(NC) PyTorch not installed — run: make install-deps"; \
			fi; \
			NUMPY=$$($(VENV_DIR)/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null); \
			if [ -n "$$NUMPY" ]; then \
				echo "  $(GREEN)✓$(NC) NumPy $$NUMPY"; \
			else \
				echo "  $(YELLOW)⚠$(NC) NumPy not installed — run: make install-deps"; \
			fi; \
			MPL=$$($(VENV_DIR)/bin/python -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null); \
			if [ -n "$$MPL" ]; then \
				echo "  $(GREEN)✓$(NC) Matplotlib $$MPL"; \
			else \
				echo "  $(YELLOW)⚠$(NC) Matplotlib not installed — run: make install-deps"; \
			fi; \
		fi; \
	else \
		echo "  $(YELLOW)⚠$(NC) No virtual environment — run: make setup-venv"; \
	fi
	@echo ""

# ===========================
# Setup targets
# ===========================

setup-venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo ""
	@echo "Activate it with:"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "Then run: make install-deps"

install-deps:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "No virtual environment found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	@echo "Installing Python dependencies..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install torch torchvision numpy matplotlib jupyter
	@echo ""
	@echo "Dependencies installed. Run 'make check-venv' to verify."

setup-cpp:
	@echo "Downloading libtorch for C++ sessions..."
	@echo ""
	@echo "For macOS (CPU):"
	@echo "  curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip -o libtorch.zip"
	@echo "  unzip libtorch.zip -d cpp/"
	@echo ""
	@echo "This will be set up in the C++ session. No action needed now."

# ===========================
# Sessions
# ===========================

list-sessions:
	@echo ""
	@echo "  ML Learning — Sessions                         Model affected"
	@echo "  ═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Fundamentals (PyCharm)"
	@echo "  ✅ 01. What is a tensor?                        Concepts only"
	@echo "  ✅ 02. Feature extraction — diffs to numbers    Concepts only"
	@echo "  ✅ 03. A single neuron                          Concepts only"
	@echo "  ✅ 04. A neural network — stacking layers       Concepts only"
	@echo "  ✅ 05. Forward pass                             Concepts only"
	@echo "  ✅ 06. Loss function                            Concepts only"
	@echo "  ✅ 6b. libtorch debug (CLion)                   Concepts only"
	@echo "  ✅ 07. Backpropagation                          Concepts only"
	@echo "  ✅ 08. Training loop                            Concepts only"
	@echo "  ✅ 09. Overfitting                              Concepts only"
	@echo ""
	@echo "  Real classifier + debug loop (PyCharm)"
	@echo "  ✅ 10.  Our classifier on real data             Our Classifier"
	@echo "  ✅ 10.5 Better features (TF-IDF + stacking)    Our Classifier"
	@echo "  ✅ 10.6 Eval harness (gate FAIL)               Our Classifier"
	@echo "  ✅ 10.7 Fix distribution mismatch (gate FAIL)  Our Classifier"
	@echo "  ✅ 10.8 Iterate to green (gate PASS)           Our Classifier"
	@echo ""
	@echo "  Deploy + pipeline integration"
	@echo "  ✅ 11.  Deploy to K8s (recap)                   Our Classifier"
	@echo "  ✅ 11.5 Pipeline signals as features            Our Classifier"
	@echo "  🔄 11.6 Deploy v5 to production                Our Classifier"
	@echo ""
	@echo "  LoRA fine-tuning"
	@echo "  ✅ 12.  LoRA concepts (toy data, learning)     GPT-2 (local)"
	@echo "  ⏳ 12b. LoRA on real corpus (production)       Ollama/Qwen"
	@echo ""
	@echo "  Run: make run SESSION=10.5"
	@echo "  Debug: open in PyCharm, set breakpoints at 🔴 lines, Debug"
	@echo "  Setup: see SETUP.md"
	@echo ""

# ===========================
# Local Ollama checks
# ===========================

OLLAMA_LOCAL_ENDPOINT := http://localhost:11434
OLLAMA_LOCAL_MODEL := qwen2.5-coder:7b
REVIEW_WORKER_DIR := ../leartech-ai-review-worker

check-ollama:
	@echo "--- Local Ollama ---"
	@if curl -s --max-time 3 "$(OLLAMA_LOCAL_ENDPOINT)/api/tags" >/dev/null 2>&1; then \
		echo "  $(GREEN)✓$(NC) Ollama server running at $(OLLAMA_LOCAL_ENDPOINT)"; \
		MODELS=$$(curl -s "$(OLLAMA_LOCAL_ENDPOINT)/api/tags" | python3 -c "import json,sys; [print('    ' + m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null); \
		if [ -n "$$MODELS" ]; then \
			echo "  $(GREEN)✓$(NC) Models available:"; \
			echo "$$MODELS"; \
		else \
			echo "  $(YELLOW)⚠$(NC) No models loaded — run: ollama pull $(OLLAMA_LOCAL_MODEL)"; \
		fi; \
		if curl -s "$(OLLAMA_LOCAL_ENDPOINT)/api/tags" | grep -q "$(OLLAMA_LOCAL_MODEL)"; then \
			echo "  $(GREEN)✓$(NC) $(OLLAMA_LOCAL_MODEL) ready (local testing model)"; \
		else \
			echo "  $(YELLOW)⚠$(NC) $(OLLAMA_LOCAL_MODEL) not found — run: ollama pull $(OLLAMA_LOCAL_MODEL)"; \
		fi; \
	else \
		echo "  $(RED)✗$(NC) Ollama not running locally"; \
		echo "    Install: brew install ollama"; \
		echo "    Start:   ollama serve"; \
		echo "    Model:   ollama pull $(OLLAMA_LOCAL_MODEL)"; \
	fi
	@echo ""

# ===========================
# Local AI Review Testing
# ===========================

review-local:
	@if [ ! -d "$(REVIEW_WORKER_DIR)" ]; then \
		echo "$(RED)✗$(NC) Review worker not found at $(REVIEW_WORKER_DIR)"; \
		echo "  Clone: git clone https://github.com/mikelear/leartech-ai-review-worker.git $(REVIEW_WORKER_DIR)"; \
		exit 1; \
	fi
	@echo "Running AI review test suite (Ollama + Claude + DeepSeek)..."
	@echo "  Ollama: $(OLLAMA_LOCAL_ENDPOINT) ($(OLLAMA_LOCAL_MODEL))"
	@echo "  APIs: Claude + DeepSeek (keys from GCP Secret Manager)"
	@echo ""
	cd $(REVIEW_WORKER_DIR) && ./run-local.sh

review-ollama:
	@if [ ! -d "$(REVIEW_WORKER_DIR)" ]; then \
		echo "$(RED)✗$(NC) Review worker not found at $(REVIEW_WORKER_DIR)"; \
		exit 1; \
	fi
	@echo "Running AI review test suite (Ollama only — no API cost)..."
	cd $(REVIEW_WORKER_DIR) && ./run-local.sh --ollama-only

review-api:
	@if [ ! -d "$(REVIEW_WORKER_DIR)" ]; then \
		echo "$(RED)✗$(NC) Review worker not found at $(REVIEW_WORKER_DIR)"; \
		exit 1; \
	fi
	@echo "Running AI review test suite (Claude + DeepSeek only — no local Ollama needed)..."
	cd $(REVIEW_WORKER_DIR) && ./run-local.sh --no-ollama

# ===========================
# Run a session from CLI
# ===========================

# Usage: make run SESSION=10.5
# Maps SESSION number to the session directory and script

SESSION_MAP_01 := sessions/01-tensors/tensors.py
SESSION_MAP_02 := sessions/02-features/features.py
SESSION_MAP_03 := sessions/03-single-neuron/single_neuron.py
SESSION_MAP_04 := sessions/04-neural-network/neural_network.py
SESSION_MAP_05 := sessions/05-forward-pass/forward_pass.py
SESSION_MAP_06 := sessions/06-loss-function/loss_function.py
SESSION_MAP_07 := sessions/07-backpropagation/backpropagation.py
SESSION_MAP_08 := sessions/08-training-loop/training_loop.py
SESSION_MAP_09 := sessions/09-overfitting/overfitting.py
SESSION_MAP_10 := sessions/10-our-classifier/classifier.py
SESSION_MAP_10.5 := sessions/10.5-better-features/better_features.py
SESSION_MAP_10.6 := sessions/10.6-eval-harness/eval_harness.py
SESSION_MAP_10.7 := sessions/10.7-fix-distribution/fix_distribution.py
SESSION_MAP_10.8 := sessions/10.8-iterate-to-green/iterate_to_green.py
SESSION_MAP_11.5 := sessions/11.5-pipeline-signals/pipeline_signals.py
SESSION_MAP_12 := sessions/12-lora-concepts/lora_concepts.py

run:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make run SESSION=10.5"; \
		echo "Available: 01-10, 10.5-10.8, 11.5"; \
		exit 1; \
	fi
	@SCRIPT="$(SESSION_MAP_$(SESSION))"; \
	if [ -z "$$SCRIPT" ]; then \
		echo "Unknown session: $(SESSION)"; \
		echo "Available: 01-10, 10.5-10.8, 11.5"; \
		exit 1; \
	fi; \
	if [ ! -f "$$SCRIPT" ]; then \
		echo "Script not found: $$SCRIPT"; \
		exit 1; \
	fi; \
	echo "Running Session $(SESSION): $$SCRIPT"; \
	echo ""; \
	$(VENV_DIR)/bin/python "$$SCRIPT"

.PHONY: help check check-python check-cpp check-ide check-gpu check-ollama check-venv setup-venv install-deps setup-cpp list-sessions run review-local review-ollama review-api _header _footer
