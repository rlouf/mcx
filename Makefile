ci:
	black --check mcx
	flake8 mcx --count --ignore=E501,E203,E731,W503,E722 --show-source --statistics
	mypy mcx/

