.PHONY: help setup install test test-cov lint format run clean notebook benchmark

help:
	@echo "SynthDetect - Makefile Commands"
	@echo "================================"
	@echo "setup          : Initial project setup (install Poetry, dependencies, spaCy)"
	@echo "install        : Install dependencies"
	@echo "test           : Run test suite"
	@echo "test-cov       : Run tests with coverage report"
	@echo "lint           : Run linting (flake8, mypy)"
	@echo "format         : Auto-format code (black, isort)"
	@echo "notebook       : Launch Jupyter notebook server"
	@echo "benchmark      : Run full benchmark suite"
	@echo "download-data  : Download datasets (HC3, M4GT)"
	@echo "build-vectordb : Build FAISS vector database"
	@echo "clean          : Remove cache and temporary files"

setup:
	pip install poetry
	poetry install --with dev,research
	poetry run python -m spacy download en_core_web_sm
	mkdir -p data/raw data/processed/features data/processed/embeddings
	mkdir -p data/vector_db data/models data/cache
	mkdir -p logs

install:
	poetry install

test:
	poetry run pytest tests/ -v -m "not slow and not integration"

test-all:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	poetry run flake8 src/ tests/ --max-line-length=100
	poetry run mypy src/ --ignore-missing-imports

format:
	poetry run black src/ tests/ scripts/
	poetry run isort src/ tests/ scripts/

notebook:
	poetry run jupyter notebook notebooks/

benchmark:
	poetry run python scripts/evaluation/benchmark.py

download-data:
	poetry run python scripts/data_preparation/download_datasets.py

build-vectordb:
	poetry run python scripts/data_preparation/build_vector_db.py

train-encoder:
	poetry run python scripts/training/train_encoder.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf data/cache/*
