PYTHON ?= $(if $(wildcard myenv/bin/python),myenv/bin/python,python3)
PIP := $(PYTHON) -m pip
STREAMLIT := $(PYTHON) -m streamlit
UVICORN := $(PYTHON) -m uvicorn
EVAL_FILE ?= evals/publishing_eval_sample.jsonl
DOC_ID ?=
USER_ID ?=
K ?= 10
QUESTION ?= What is this document about?
CATALOG_OUT ?=
ROW_OUT ?=
CATALOG_FILE ?=
REVIEW_MD ?=
REVIEW_JSON ?=
OUT_FILE ?=
POOL_K ?= 40
COMPARE_RERANK ?= false

run:
	$(STREAMLIT) run app.py
api:
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000
install:
	$(PIP) install -r requirements.txt
lint:
	ruff check .
format:
	black .
test:
	$(PYTHON) -m pytest
eval-retrieval:
	$(PYTHON) scripts/evaluate_retrieval.py --eval-file $(EVAL_FILE) --user-id "$(USER_ID)" --k $(K) $(if $(filter true,$(COMPARE_RERANK)),--compare-rerank --candidate-pool-k $(POOL_K),)
eval-rerank-impact:
	$(PYTHON) scripts/evaluate_retrieval.py --eval-file $(EVAL_FILE) --user-id "$(USER_ID)" --k $(K) --compare-rerank --candidate-pool-k $(POOL_K)
eval-prepare-label:
	@if [ -z "$(DOC_ID)" ]; then echo "DOC_ID is required. Example: make eval-prepare-label DOC_ID=c385eadd61 QUESTION='Who is Dorothy?'"; exit 1; fi
	$(PYTHON) scripts/prepare_retrieval_labels.py \
		--doc-id "$(DOC_ID)" \
		--question "$(QUESTION)" \
		--user-id "$(USER_ID)" \
		--k $(K) \
		$(if $(CATALOG_OUT),--catalog-out "$(CATALOG_OUT)",) \
		$(if $(ROW_OUT),--row-out "$(ROW_OUT)",)
eval-fill-retrieved:
	@if [ -z "$(DOC_ID)" ]; then echo "DOC_ID is required. Example: make eval-fill-retrieved DOC_ID=c385eadd61"; exit 1; fi
	$(PYTHON) scripts/fill_retrieved_chunk_ids.py \
		--eval-file $(EVAL_FILE) \
		--doc-id "$(DOC_ID)" \
		--user-id "$(USER_ID)" \
		--k $(K) \
		--verbose \
		$(if $(OUT_FILE),--out-file "$(OUT_FILE)",)
eval-review-set:
	@if [ -z "$(DOC_ID)" ]; then echo "DOC_ID is required. Example: make eval-review-set DOC_ID=c385eadd61"; exit 1; fi
	$(PYTHON) scripts/build_eval_review_bundle.py \
		--eval-file $(EVAL_FILE) \
		--doc-id "$(DOC_ID)" \
		--user-id "$(USER_ID)" \
		$(if $(CATALOG_FILE),--catalog-file "$(CATALOG_FILE)",) \
		$(if $(REVIEW_MD),--out-md "$(REVIEW_MD)",) \
		$(if $(REVIEW_JSON),--out-json "$(REVIEW_JSON)",)
eval-review-bundle: eval-review-set
eval-review-checklist:
	@if [ -z "$(DOC_ID)" ]; then echo "DOC_ID is required. Example: make eval-review-checklist DOC_ID=c385eadd61"; exit 1; fi
	$(STREAMLIT) run scripts/review_retrieval_checklist.py -- \
		--bundle-file "$(if $(REVIEW_JSON),$(REVIEW_JSON),evals/review_bundle_$(DOC_ID).json)" \
		--eval-file "$(EVAL_FILE)"
dev:
	docker compose up --build
dev-up:
	docker compose up --force-recreate
dev-down:
	docker compose down
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
