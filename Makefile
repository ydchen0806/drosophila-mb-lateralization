PYTHON ?= python3

.PHONY: validate figures test reproduce clean

validate:
	$(PYTHON) scripts/validate_source_data.py

figures:
	$(PYTHON) scripts/build_figures.py
	$(PYTHON) scripts/build_external_validation.py

test:
	$(PYTHON) -m pytest -q

reproduce: validate figures test

clean:
	rm -f figures/*.pdf figures/*.png figures/*.json
