import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import build_parser


parser = build_parser("Compatibility wrapper for the benchmark plotting workflow")
parser.parse_args()

subprocess.run([sys.executable, "scripts/bench_all.py"], check=True)
