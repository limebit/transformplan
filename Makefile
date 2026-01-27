VENV_NAME ?= .venv

ifeq ($(OS),Windows_NT)
    USER_PYTHON ?= python
    VENV_BIN := $(VENV_NAME)/Scripts
    rmrf = rmdir /s /q
    rmf = del /q
else
    USER_PYTHON ?= python3
    VENV_BIN := $(VENV_NAME)/bin
    rmrf = rm -rf
    rmf = rm -f
endif

VENV_PYTHON := $(VENV_BIN)/python
VENV_UV := $(VENV_BIN)/uv
UV_LOC := $(shell $(USER_PYTHON) -c "import shutil; print(shutil.which('uv') if shutil.which('uv') else '$(VENV_UV)')")

.PHONY: prepare-venv install install-dev test lint format clean

.DEFAULT_GOAL := install-dev

prepare-venv: $(VENV_NAME)

$(VENV_NAME):
ifeq ($(UV_LOC),$(VENV_UV))
	@echo "Using .venv installed uv: $(UV_LOC)"
	$(MAKE) clean && $(USER_PYTHON) -m venv $(VENV_NAME)
	$(VENV_PYTHON) -m pip install -U pip
	$(VENV_PYTHON) -m pip install uv
else
	@echo "Using global uv: $(UV_LOC)"
	$(MAKE) clean && $(UV_LOC) venv $(VENV_NAME)
endif

install: prepare-venv
	$(UV_LOC) sync
	$(UV_LOC) pip install -e .

install-dev: prepare-venv
	$(UV_LOC) sync --group dev
	$(UV_LOC) pip install -e .
	$(UV_LOC) run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout

test: install-dev
	$(UV_LOC) run pytest -vv -W error

lint: install-dev
	$(UV_LOC) run ruff check
	$(UV_LOC) run ruff check --select I
	$(UV_LOC) run python -m pyright

format: install-dev
	$(UV_LOC) run ruff check --select I --fix
	$(UV_LOC) run ruff format

clean:
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_NAME) $(rmrf) $(VENV_NAME)
	@for %%d in (.ruff_cache .pytest_cache transformplan.egg-info) do @if exist %%d $(rmrf) %%d
	@if exist .vscode\*.log $(rmf) .vscode\*.log
	@for /d /r %%i in (__pycache__) do @if exist "%%i" $(rmrf) "%%i"
else
	$(rmrf) $(VENV_NAME) .ruff_cache .pytest_cache transformplan.egg-info
	$(rmf) .vscode/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
endif
