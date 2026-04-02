import argparse
import re
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import REPO_ROOT, resolve_dataset_path


RESULT_RE = re.compile(r"Final result:\s*(\d+)")
VARIANT_MATRIX = {
    "q11": [
        ["./bin/mod_q11_sycldb", "-r", "1"],
        ["./bin/adp_q11_sycldb", "-r", "1"],
        ["./bin/hrd_q11_sycldb", "-r", "1"],
        ["./bin/mrd_q11", "-r", "1"],
    ],
    "q21": [
        ["./bin/mod_q21_sycldb", "-r", "1"],
        ["./bin/adp_q21_sycldb", "-r", "1"],
        ["./bin/hrd_q21_sycldb", "-r", "1"],
        ["./bin/mrd_q21", "-r", "1"],
    ],
}


def run_and_extract(command: list[str]) -> int:
    proc = subprocess.run(command, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    match = RESULT_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse final result from {command[0]}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify benchmark result consistency across variants")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["q11", "q21"],
        choices=sorted(VARIANT_MATRIX.keys()),
    )
    args = parser.parse_args()

    dataset = resolve_dataset_path(args.dataset)

    for query in args.queries:
        commands = []
        for command in VARIANT_MATRIX[query]:
            binary_path = REPO_ROOT / command[0][2:]
            if not binary_path.exists():
                raise FileNotFoundError(f"Missing benchmark binary: {binary_path}")
            commands.append(command + ["-p", dataset])

        baseline = None
        print(f"Verifying {query}...")
        for command in commands:
            result = run_and_extract(command)
            print(f"  {command[0]} -> {result}")
            if baseline is None:
                baseline = result
            elif result != baseline:
                raise RuntimeError(
                    f"Result mismatch for {query}: expected {baseline}, got {result} from {command[0]}"
                )

    print("All selected variants produced matching final results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
