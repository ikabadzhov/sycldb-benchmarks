import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import (
    REPO_ROOT,
    build_parser,
    resolve_dataset_path,
    resolve_tool_path,
)


PATTERNS = ["standard", "coalesced", "tiled"]
QUERIES = ["q11", "q21"]
MODELS = ["Modular", "JIT Fusion", "Hardcoded"]
PATTERN_SUFFIX = {
    "standard": "sycldb",
    "coalesced": "sycldbcoalesced",
    "tiled": "sycldbtiled",
}
MODEL_PREFIX = {"Modular": "mod", "JIT Fusion": "adp", "Hardcoded": "hrd"}
MODEL_FILENAME = {
    "Modular": "modular",
    "JIT Fusion": "adaptive",
    "Hardcoded": "hardcoded",
}
RUN_TIME_RE = re.compile(r"(?:Run|Iteration)\s+\d+:\s*([\d\.]+)\s*ms")


def needs_rebuild(output_path, source_path):
    if not output_path.exists():
        return True
    return source_path.stat().st_mtime > output_path.stat().st_mtime


def parse_benchmark_output(stdout: str) -> list[float]:
    return [float(match) for match in RUN_TIME_RE.findall(stdout)]


def summarize_times(times_ms: list[float]) -> tuple[float | None, float | None]:
    if not times_ms:
        return None, None
    avg = statistics.fmean(times_ms)
    variance = statistics.fmean([(sample - avg) * (sample - avg) for sample in times_ms])
    return avg, variance ** 0.5


def resolve_sycl_source(query: str, pattern: str, model: str) -> Path:
    return Path("src") / pattern / f"{query}_{MODEL_FILENAME[model]}.cpp"


def resolve_binary_name(query: str, pattern: str, model: str) -> str:
    return f"{MODEL_PREFIX[model]}_{query}_{PATTERN_SUFFIX[pattern]}"


def run_bench(cmd):
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error executing {cmd[0]}: {stderr}")
        return None

    times_ms = parse_benchmark_output(stdout)
    if times_ms:
        return times_ms

    print(f"No per-run timings found in output for {cmd[0]}:\n{stdout}")
    return None


def build_compile_commands(acpp_path: str, nvcc_path: str) -> list[list[str]]:
    compile_cmds = []
    for query in QUERIES:
        for pattern in PATTERNS:
            for model in MODELS:
                output = REPO_ROOT / "bin" / resolve_binary_name(query, pattern, model)
                source = REPO_ROOT / resolve_sycl_source(query, pattern, model)
                if needs_rebuild(output, source):
                    compile_cmds.append(
                        [
                            acpp_path,
                            "-O3",
                            "-std=c++20",
                            "--acpp-targets=generic",
                            str(source),
                            "-o",
                            str(output),
                        ]
                    )

    utils_source = REPO_ROOT / "src" / "utils" / "sycl_ls.cpp"
    utils_output = REPO_ROOT / "bin" / "sycl_ls"
    if needs_rebuild(utils_output, utils_source):
        compile_cmds.append(
            [
                acpp_path,
                "-O3",
                "-std=c++20",
                "--acpp-targets=generic",
                str(utils_source),
                "-o",
                str(utils_output),
            ]
        )

    for query in ("q11", "q21"):
        output = REPO_ROOT / "bin" / f"mrd_{query}"
        source = REPO_ROOT / "src" / "cuda" / f"{query}_mordred.cu"
        if needs_rebuild(output, source):
            compile_cmds.append(
                [
                    nvcc_path,
                    "-O3",
                    "-arch=native",
                    str(source),
                    "-o",
                    str(output),
                ]
            )
    return compile_cmds


def main() -> int:
    parser = build_parser("Compile and run benchmark variants")
    args = parser.parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    acpp_path = resolve_tool_path(args.acpp, "SYCLDB_ACPP")
    nvcc_path = resolve_tool_path(args.nvcc, "SYCLDB_NVCC")
    repetitions = args.repetitions
    device_id = args.device

    results = {
        "Q1.1": {"JIT Fusion": [], "Modular": [], "Hardcoded": [], "CUDA": []},
        "Q2.1": {"JIT Fusion": [], "Modular": [], "Hardcoded": [], "CUDA": []},
    }
    raw_data = []

    os.makedirs(REPO_ROOT / "bin", exist_ok=True)
    os.makedirs(REPO_ROOT / "results", exist_ok=True)

    compile_cmds = build_compile_commands(acpp_path, nvcc_path)
    if compile_cmds:
        print("Compiling missing variants...")
        for cmd in compile_cmds:
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    for query in QUERIES:
        query_key = "Q1.1" if query == "q11" else "Q2.1"
        for pattern in PATTERNS:
            for model_key in MODELS:
                cmd = [
                    str(REPO_ROOT / "bin" / resolve_binary_name(query, pattern, model_key)),
                    "-r",
                    str(repetitions),
                    "-p",
                    dataset_path,
                ]
                if device_id is not None:
                    cmd.extend(["-d", str(device_id)])
                times_ms = run_bench(cmd)
                results[query_key][model_key].append(times_ms)
                raw_data.append(
                    {
                        "query": query,
                        "variant": PATTERN_SUFFIX[pattern],
                        "model": model_key,
                        "times_ms": times_ms or [],
                    }
                )

    for query_key, query in (("Q1.1", "q11"), ("Q2.1", "q21")):
        times_ms = run_bench(
            [str(REPO_ROOT / "bin" / f"mrd_{query}"), "-r", str(repetitions), "-p", dataset_path]
        )
        results[query_key]["CUDA"].append(times_ms)
        raw_data.append(
            {
                "query": query,
                "variant": "mordred",
                "model": "CUDA",
                "times_ms": times_ms or [],
            }
        )

    with open(REPO_ROOT / "results" / "benchmark_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)

    print("\n| Query | Strategy | Hardcoded (ms) | JIT Fusion (ms) | Modular (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for query in ["Q1.1", "Q2.1"]:
        for i, strategy in enumerate(["Standard", "Coalesced", "Tiled"]):
            try:
                h_avg, _ = summarize_times(results[query]["Hardcoded"][i])
                a_avg, _ = summarize_times(results[query]["JIT Fusion"][i])
                m_avg, _ = summarize_times(results[query]["Modular"][i])
                if h_avg is None or a_avg is None or m_avg is None:
                    raise TypeError
                print(f"| {query} | {strategy} | {h_avg:.3f} | {a_avg:.3f} | {m_avg:.3f} |")
            except (TypeError, IndexError):
                print(f"| {query} | {strategy} | N/A | N/A | N/A |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
