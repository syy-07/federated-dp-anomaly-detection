.PHONY: test lint type format
test:
	pytest -q
lint:
	flake8 src
type:
	mypy src || true
format:
	python -m pip install black && black .
