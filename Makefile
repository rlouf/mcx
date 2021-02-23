lint-checks:
	isort --profile black --check mcx tests
	black --check mcx tests
	flake8 mcx tests --count --ignore=E501,E203,E731,W503,E722 --show-source --statistics
	mypy mcx/

test:
	pytest -n 4 tests

