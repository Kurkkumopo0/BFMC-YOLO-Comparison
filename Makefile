.PHONY: init train

GPU ?= 1
VENV_DIR:= .venv
MODEL ?=
DATA ?=
EPOCHS ?=
RUN_NAME ?=
TRAIN_SCRIPT := python/train.py

AVAILABLE_MODELS := yolo11n.pt yolo11s.pt yolov8n.pt yolov11s.pt
AVAILABLE_DATASETS := MNIST CIFAR10 ImageNet

init: $(VENV_DIR)
	@echo "Installing dependencies"
	@$(VENV_DIR)/bin/pip install -r requirements.txt
ifeq ($(GPU), 1)
	@echo "Installing PyTorch with CUDA support"
	@echo "Detecting CUDA version..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		CUDA_VERSION=$$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1); \
		CUDA_VERSION_TRIMMED=$${CUDA_VERSION//./}; \
		echo "Detected CUDA Version: $$CUDA_VERSION (using cu$$CUDA_VERSION_TRIMMED)"; \
		$(VENV_DIR)/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu$$CUDA_VERSION_TRIMMED;
	else \
		echo "Warning: nvidia-smi not found. Installing CUDA 12.6 build by default"; \
		$(VENV_DIR)/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126;
	@echo "Verifying installation"
	@python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
else
	@echo "Step 3: Installing PyTorch CPU-only version"
	@$(VENV_DIR)/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "Verifying installation"
	@python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('CPU-only installation')"
endif

$(VENV_DIR):
	@echo "Creating virtual environment at $(VENV_DIR)/bin/activate"
	@python3 -m venv $(VENV_DIR)



train:
	@echo "Starting Training Configuration"
	@MODEL="$(MODEL)"; DATA="$(DATA)"; EPOCHS="$(EPOCHS)"; RUN_NAME="$(RUN_NAME)"; \
	if [ -z "$$MODEL" ]; then \
		echo "\nAvailable Models:"; \
		i=1; for m in $(AVAILABLE_MODELS); do echo "$$i) $$m"; i=$$((i+1)); done; \
		printf "Select model (1-$(words $(AVAILABLE_MODELS))): "; \
		read sel; \
		MODEL=$$(echo "$(AVAILABLE_MODELS)" | awk -v sel=$$sel '{print $$sel}'); \
		if [ -z "$$MODEL" ]; then \
			MODEL="$(word 1,$(AVAILABLE_MODELS))"; \
			echo "Invalid selection. Using default: $$MODEL"; \
		fi; \
	fi; \
	if [ -z "$$DATA" ]; then \
		echo "\nAvailable Datasets:"; \
		i=1; for d in $(AVAILABLE_DATASETS); do echo "$$i) $$d"; i=$$((i+1)); done; \
		printf "Select dataset (1-$(words $(AVAILABLE_DATASETS))): "; \
		read sel; \
		DATA=$$(echo "$(AVAILABLE_DATASETS)" | awk -v sel=$$sel '{print $$sel}'); \
		if [ -z "$$DATA" ]; then \
			DATA="$(word 1,$(AVAILABLE_DATASETS))"; \
			echo "Invalid selection. Using default: $$DATA"; \
		fi; \
	fi; \
	if [ -z "$$EPOCHS" ]; then \
		printf "\nEnter number of epochs (default: 10): "; \
		read sel; \
		if [ -z "$$sel" ]; then \
			EPOCHS="10"; \
		else \
			EPOCHS="$$sel"; \
		fi; \
	fi; \
	if [ -z "$$RUN_NAME" ]; then \
		printf "\nEnter run RUN_NAME (default: run-$$(date +%Y%m%d-%H%M)): "; \
		read sel; \
		if [ -z "$$sel" ]; then \
			RUN_NAME="run-$$(date +%Y%m%d-%H%M%S)"; \
		else \
			RUN_NAME="$$sel"; \
		fi; \
	fi; \
	echo "executing python3 $(TRAIN_SCRIPT) -m $$MODEL -d $$DATA -e $$EPOCHS -n $$RUN_NAME"; \
	python3 $(TRAIN_SCRIPT) -m $$MODEL -d $$DATA -e $$EPOCHS -n $$RUN_NAME; \
