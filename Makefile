run:
	streamlit run app.py
install:
	pip install -r requirements.txt
lint:
	ruff check .
format:
	black .
test:
	pytest
dev:
	docker compose up --build
dev-up:
	docker compose up --force-recreate
dev-down:
	docker compose down
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
