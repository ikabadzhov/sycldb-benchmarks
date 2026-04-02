.PHONY: help benchmark verify plot-measured list-devices

help:
	@echo "Targets: benchmark verify plot-measured list-devices"

list-devices:
	@mkdir -p bin
	@"$$(python3 -c "from scripts.benchmark_config import resolve_tool_path; print(resolve_tool_path(None, 'SYCLDB_ACPP'))")" -O3 -std=c++20 --acpp-targets=generic src/utils/sycl_ls.cpp -o bin/sycl_ls
	@./bin/sycl_ls

benchmark:
	python3 scripts/bench_all.py

verify:
	python3 scripts/verify_results.py

plot-measured:
	python3 scripts/plot_measured.py
