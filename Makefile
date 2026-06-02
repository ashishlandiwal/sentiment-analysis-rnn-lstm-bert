.PHONY: install extras dev test lint compare compare-fast docker-build docker-run

install:
	pip install -r requirements.txt

extras:
	pip install -r requirements-optional.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

# Full comparison (RNN + LSTM + DistilBERT). Needs `make extras`.
compare:
	PYTHONPATH=src python -m sentiment.compare --output reports

# RNN + LSTM only (still needs the dataset, but no transformer download).
compare-fast:
	PYTHONPATH=src python -m sentiment.compare --skip-bert --output reports

docker-build:
	docker build -t sentiment-analysis .

docker-run:
	docker run --rm sentiment-analysis
