.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache

fclean: clean
	rm -rf .venv

re: fclean install

lint:
	uv run flake8 --exclude=.venv,.git,__pycache__,llm_sdk .
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --explicit-package-bases --follow-imports=silent

lint-strict:
	uv run flake8 --exclude=.venv,.git,__pycache__,llm_sdk .
	uv run mypy src/ --strict --explicit-package-bases --follow-imports=silent