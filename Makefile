PY ?= python
VENV := .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif
export PYTHONPATH := .

.DEFAULT_GOAL := help

.PHONY: help
help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------------

.PHONY: install
install: ## create the venv and install everything
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip
	$(BIN)/python -m pip install -e ".[dev,llm]"

# ---------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------

.PHONY: ingest
ingest: ## generate the corpus and build the index (idempotent)
	$(BIN)/python -m autopsy.cli ingest

.PHONY: fetch
fetch: ## download the real documentation sources (licence-gated)
	$(BIN)/python -m corpus.build fetch

# ---------------------------------------------------------------------------------
# the three artifacts
# ---------------------------------------------------------------------------------

.PHONY: eval
eval: ## run every suite headless; exits non-zero on a NEW failure
	$(BIN)/python -m autopsy.cli eval

.PHONY: eval-baseline
eval-baseline: ## record the current failure set as the reference
	$(BIN)/python -m autopsy.cli eval --update-baseline

.PHONY: ablate
ablate: ## the counterfactual sweep -> reports/ablation.md
	$(BIN)/python -m autopsy.cli ablate -n 220

.PHONY: ablate-snapshot
ablate-snapshot: ## pin the current outcome distribution as the regression reference
	$(BIN)/python -m autopsy.cli ablate -n 220 --snapshot

.PHONY: calibrate
calibrate: ## judge calibration -> reports/judge-calibration.md
	$(BIN)/python -m autopsy.cli calibrate --derive -n 120

.PHONY: calibrate-gate
calibrate-gate: ## derive gate.reads and gate.threshold from the corpus
	$(BIN)/python -m autopsy.cli calibrate-gate

.PHONY: reports
reports: eval ablate calibrate ## regenerate everything under reports/

# ---------------------------------------------------------------------------------
# dev
# ---------------------------------------------------------------------------------

.PHONY: test
test: ## unit and property tests
	$(BIN)/python -m pytest tests -q

.PHONY: schema
schema: ## regenerate the JSON Schema and the frontend's Zod types
	$(BIN)/python -m autopsy.cli schema

.PHONY: demo-traces
demo-traces: ## freeze pre-recorded traces for the keyless demo
	$(BIN)/python -m autopsy.cli demo

.PHONY: api
api: ## run the inspector API on :8000
	$(BIN)/python -m uvicorn api.main:app --reload --port 8000

.PHONY: web
web: ## run the inspector UI on :5173
	cd web && npm install && npm run dev

.PHONY: demo
demo: ## docker compose: api + web
	docker compose up --build

.PHONY: ci
ci: test eval ## what the pipeline runs
