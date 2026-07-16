# Makefile — Health Monitor project
# Targets for data generation, training, testing, deployment, and cleanup.
# Works with GNU Make on Windows (via MSYS2/MinGW, WSL, or Git Bash)
# and on Linux/macOS.

PYTHON      ?= python
PIP         ?= pip
PYTEST      ?= pytest
VENV_DIR    := venv
DATA_DIR    := data/synthetic/raw
MODELS_DIR  := models
SCRIPTS_DIR := scripts
SRC_DIR     := src
HF_DIR      := hf_space

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows:  $(VENV_DIR)\\Scripts\\activate"
	@echo "  Unix:     source $(VENV_DIR)/bin/activate"

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------
.PHONY: data
data:
	$(PYTHON) $(SCRIPTS_DIR)/generate_synthetic.py \
		--num-sessions 100 \
		--output-dir $(DATA_DIR) \
		--seed 42 \
		--include-labels

.PHONY: data-quick
data-quick:
	$(PYTHON) $(SCRIPTS_DIR)/generate_synthetic.py \
		--num-sessions 10 \
		--duration 30 \
		--output-dir $(DATA_DIR) \
		--seed 42 \
		--include-labels

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
.PHONY: test
test:
	$(PYTEST) tests/ -v

.PHONY: test-e2e
test-e2e:
	$(PYTHON) $(SCRIPTS_DIR)/e2e_test.py

# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------
.PHONY: train
train:
	$(PYTHON) $(SCRIPTS_DIR)/train_pipeline.py \
		--num-sessions 30 \
		--no-gpu \
		--seed 42

.PHONY: train-quick
train-quick:
	$(PYTHON) $(SCRIPTS_DIR)/train_pipeline.py --quick-test

.PHONY: train-full
train-full:
	$(PYTHON) $(SCRIPTS_DIR)/train_pipeline.py \
		--num-sessions 100 \
		--no-gpu \
		--seed 42

# ---------------------------------------------------------------------------
# Validate (quick smoke test of the whole pipeline)
# ---------------------------------------------------------------------------
.PHONY: validate
validate:
	$(PYTHON) $(SCRIPTS_DIR)/quick_validate.py

# ---------------------------------------------------------------------------
# HF Space deployment helpers
# ---------------------------------------------------------------------------
.PHONY: deploy-check
deploy-check:
	@echo "=== HF Space Deployment Checklist ==="
	@echo "[1] hf_space/app.py         : $$(if exist $(HF_DIR)\\app.py (echo OK) else echo MISSING)"
	@echo "[2] hf_space/requirements.txt: $$(if exist $(HF_DIR)\\requirements.txt (echo OK) else echo MISSING)"
	@echo "[3] hf_space/README.md      : $$(if exist $(HF_DIR)\\README.md (echo OK) else echo MISSING)"
	@echo "[4] models/*.joblib         : $$(if exist $(MODELS_DIR)\\cardiac.joblib (echo OK) else echo MISSING)"
	@echo "[5] models/feature_names.json: $$(if exist $(MODELS_DIR)\\feature_names.json (echo OK) else echo MISSING)"
	@echo "[6] src/ directory          : $$(if exist $(SRC_DIR)\\__init__.py (echo OK) else echo MISSING)"
	@echo "[7] data/synthetic/ directory: $$(if exist data\\synthetic\\generator.py (echo OK) else echo MISSING)"
	@echo ""
	@echo "To deploy:"
	@echo "  1. Create a Space at https://huggingface.co/new-space"
	@echo "     - SDK: Gradio"
	@echo "     - Space name: health-monitor"
	@echo "  2. Clone the Space locally"
	@echo "  3. Copy these files into the Space directory:"
	@echo "     - hf_space/app.py (rename to app.py)"
	@echo "     - hf_space/requirements.txt"
	@echo "     - hf_space/README.md"
	@echo "     - Entire src/ directory"
	@echo "     - Entire models/ directory (joblib files)"
	@echo "     - data/synthetic/generator.py"
	@echo "  4. Git add, commit, push"

.PHONY: deploy-prepare
deploy-prepare:
	$(PYTHON) $(SCRIPTS_DIR)/prepare_deploy.py

.PHONY: deploy-run
deploy-run:
	$(PYTHON) $(HF_DIR)/app.py

# ---------------------------------------------------------------------------
# Clean cache and temp files
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	# Python caches
	-for /d %%i in ($(SRC_DIR)\__pycache__) do rd /s /q "%%i" 2>nul
	-for /d %%i in ($(SCRIPTS_DIR)\__pycache__) do rd /s /q "%%i" 2>nul
	-del /s /q *.pyc 2>nul
	-del /s /q *.pyo 2>nul
	-del /s /q *.egg-info 2>nul
	# Temp files
	-del /s /q *.tmp 2>nul
	-del /s /q *.log 2>nul
	-del /s /q Thumbs.db 2>nul
	@echo "Cleaned cache and temp files."

.PHONY: clean-models
clean-models:
	-del /q $(MODELS_DIR)\\*.joblib 2>nul
	-del /q $(MODELS_DIR)\\*.onnx 2>nul
	-del /q $(MODELS_DIR)\\*.json 2>nul
	@echo "Cleaned model files."

.PHONY: clean-data
clean-data:
	-del /q $(DATA_DIR)\\*.json 2>nul
	@echo "Cleaned synthetic data."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  venv          Create Python virtual environment"
	@echo "  install       Install dependencies from requirements.txt"
	@echo "  data          Generate 100 synthetic sessions (5 min each)"
	@echo "  data-quick    Generate 10 quick synthetic sessions (30 s each)"
	@echo "  test          Run pytest test suite"
	@echo "  test-e2e      Run end-to-end inference validation"
	@echo "  train         Train models (30 sessions, CPU)"
	@echo "  train-quick   Quick training (10 sessions, 2 boost rounds)"
	@echo "  train-full    Full training (100 sessions, CPU)"
	@echo "  validate      Quick smoke test of inference pipeline"
	@echo "  deploy-check  Check readiness for HF Space deployment"
	@echo "  deploy-prepare Prepare deployment package"
	@echo "  deploy-run    Run Gradio app locally (hf_space/app.py)"
	@echo "  clean         Remove Python caches and temp files"
	@echo "  clean-models  Remove all trained model files"
	@echo "  clean-data    Remove all synthetic data files"
	@echo "  help          Show this message"
