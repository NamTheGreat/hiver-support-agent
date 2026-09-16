.PHONY: all test run clean figures

PYTHON := python3

all:
	$(PYTHON) run_all.py

subsample:
	$(PYTHON) run_all.py --prefer-subsample

test:
	pytest tests/ -v -x

figures:
	$(PYTHON) -m eval.generate_figures

clean:
	rm -rf .cache reports/figures/*.png reports/results/*.json reports/results/*.csv reports/judge_agreement.json
