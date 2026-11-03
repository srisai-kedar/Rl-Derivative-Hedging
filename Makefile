.PHONY: install lint type test test-fast audit train eval dashboard
install:
	pip install -r requirements.txt -r requirements-dev.txt hypothesis && pip install -e .
lint:
	ruff check src tests
type:
	mypy src
test:
	pytest tests/ -m "not slow" --ignore=tests/test_hpo.py --cov=src --cov-fail-under=85
test-fast:
	pytest tests/test_env_hardening.py tests/test_uncertainty.py tests/test_bs_properties.py -q
audit:
	pip-audit -r requirements.txt
train:
	python -m src.training.train --config configs/training.yaml
dashboard:
	streamlit run src/dashboard/app.py
