import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import build_parser


parser = build_parser("Compatibility wrapper for the measured benchmark plotting workflow")
args = parser.parse_args()

cmd = [sys.executable, "scripts/bench_all.py"]
if args.dataset:
    cmd.extend(["--dataset", args.dataset])
if args.acpp:
    cmd.extend(["--acpp", args.acpp])
if args.nvcc:
    cmd.extend(["--nvcc", args.nvcc])
if args.device is not None:
    cmd.extend(["--device", str(args.device)])
cmd.extend(["--repetitions", str(args.repetitions)])
subprocess.run(cmd, check=True)
subprocess.run([sys.executable, "scripts/plot_measured.py"], check=True)
