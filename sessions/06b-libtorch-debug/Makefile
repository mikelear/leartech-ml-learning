# libtorch Debug Session — Setup & Build
#
# Usage:
#   make setup    Download and extract libtorch (one time only)
#   make build    Build the C++ debug binary
#   make run      Run the binary
#   make debug    Build and run with lldb (terminal debugger)
#   make clean    Remove build artifacts

LIBTORCH_URL := https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip
LIBTORCH_DIR := libtorch
BUILD_DIR := build

help:
	@echo ""
	@echo "  libtorch Debug Session"
	@echo "  ======================"
	@echo ""
	@echo "  First time:  make setup    (downloads libtorch ~60MB)"
	@echo "  Then:        make build    (compiles the C++ binary)"
	@echo "  Run:         make run      (runs the binary)"
	@echo "  Debug:       make debug    (runs with lldb debugger)"
	@echo ""
	@echo "  For CLion: just open this folder, it detects CMakeLists.txt"
	@echo "  Set breakpoints on 🔴 lines in main.cpp and hit Debug"
	@echo ""

setup:
	@if [ -d "$(LIBTORCH_DIR)" ]; then \
		echo "libtorch already downloaded"; \
	else \
		echo "Downloading libtorch (ARM64 macOS, ~60MB)..."; \
		curl -L "$(LIBTORCH_URL)" -o libtorch.zip; \
		unzip -q libtorch.zip; \
		rm libtorch.zip; \
		echo "libtorch extracted to $(LIBTORCH_DIR)/"; \
	fi

build: setup
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Debug && make
	@echo "Built: $(BUILD_DIR)/libtorch_debug"

run: build
	@cd $(BUILD_DIR) && ./libtorch_debug

debug: build
	@cd $(BUILD_DIR) && lldb ./libtorch_debug

clean:
	rm -rf $(BUILD_DIR)

.PHONY: help setup build run debug clean
