run:
	streamlit run app.py
install:
	pip install -r requirements.txt
lint:
	pylint src
format:
	black .
test:
	pytest
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
