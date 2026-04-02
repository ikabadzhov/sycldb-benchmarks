from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CANDIDATES = [
    "/media/ssb/sf100_columnar",
    "/media/ssb/s100_columnar",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_path: str | None = None
    acpp_path: str | None = None
    nvcc_path: str | None = None
    repetitions: int = 10
    bin_dir: str = str(REPO_ROOT / "bin")
    results_dir: str = str(REPO_ROOT / "results")


def resolve_dataset_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_value = os.environ.get("SYCLDB_SSB_PATH")
    if env_value:
        return env_value
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_DATASET_CANDIDATES[0]


def resolve_tool_path(explicit: str | None, env_var: str) -> str:
    if explicit:
        return explicit
    return os.environ.get(env_var, env_var.lower())


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--acpp", default=None)
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("-r", "--repetitions", type=int, default=10)
    return parser
