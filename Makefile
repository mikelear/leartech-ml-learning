# ML Learning Journey — Pre-Session Checks & Setup
#
# Run 'make' to see all available targets
# Run 'make check' before each session to verify your environment

PYTHON := python3
PIP := pip3
VENV_DIR := .venv
CLUSTER_CONTEXT := modern-burro-admin

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
	@echo "  Setup:"
	@echo "    check            Run all pre-session checks"
	@echo "    check-python     Check Python and pip"
	@echo "    check-cpp        Check C++ toolchain (clang, cmake)"
	@echo "    check-ide        Check PyCharm and CLion"
	@echo "    check-gpu        Check Azure GPU node and Ollama"
	@echo "    check-venv       Check virtual environment and packages"
	@echo ""
	@echo "    setup-venv       Create Python virtual environment"
	@echo "    install-deps     Install Python dependencies (PyTorch, numpy, etc.)"
	@echo "    setup-cpp        Download libtorch for C++ sessions"
	@echo ""
	@echo "  Sessions:"
	@echo "    list-sessions    Show all sessions and progress"
	@echo ""
	@echo "  Run 'make check' before your first session."
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
	@echo "  Session Plan"
	@echo "  ============"
	@echo ""
	@echo "  Track 1: Python (PyCharm)"
	@echo "  ─────────────────────────"
	@echo "  01. What is a tensor?"
	@echo "  02. Feature extraction — diffs to numbers"
	@echo "  03. A single neuron"
	@echo "  04. A neural network — stacking layers"
	@echo "  05. Forward pass"
	@echo "  06. Loss function"
	@echo "  07. Backpropagation"
	@echo "  08. Training loop"
	@echo "  09. Overfitting"
	@echo "  10. Our classifier — PASS/FAIL on real diffs"
	@echo "  11. Deploy it — Flask API + K8s"
	@echo "  12. LoRA concepts — what fine-tuning changes"
	@echo ""
	@echo "  Track 2: C++ (CLion)"
	@echo "  ─────────────────────"
	@echo "  03c. Matrix multiply under the hood"
	@echo "  05c. Step through a forward pass"
	@echo "  07c. Backprop — see the chain rule"
	@echo "  08c. SGD weight update in raw memory"
	@echo ""

.PHONY: help check check-python check-cpp check-ide check-gpu check-venv setup-venv install-deps setup-cpp list-sessions _header _footer
