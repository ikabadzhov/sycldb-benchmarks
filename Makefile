.PHONY: help benchmark verify plot-measured

help:
	@echo "Targets: benchmark verify plot-measured"

benchmark:
	python3 scripts/bench_all.py

verify:
	python3 scripts/verify_results.py

plot-measured:
	python3 scripts/plot_measured.py
