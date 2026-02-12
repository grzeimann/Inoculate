# Convenience Makefile mirroring common local tasks

.PHONY: help install dev docs test clean

help:
	@echo "Common targets:"
	@echo "  make install   - install package"
	@echo "  make dev       - install package with dev+docs extras"
	@echo "  make docs      - build Sphinx HTML docs"
	@echo "  make test      - run pytest"
	@echo "  make clean     - remove build and docs artifacts"

install:
	pip install -U .

dev:
	pip install -U .[dev,docs]

docs:
	SPHINX_THEME=sphinx_rtd_theme sphinx-build -b html docs docs/_build/html

test:
	pytest

clean:
	rm -rf build dist *.egg-info
	rm -rf docs/_build
