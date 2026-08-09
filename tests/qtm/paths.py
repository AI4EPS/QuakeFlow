"""Locate the workflow these tests exercise.

The scripts under test live in scripts/ and read scripts/local/<region>/, so paths are
resolved from this file rather than the working directory -- the tests can then be run from
anywhere, which the old scripts/benchmark_qtm/ layout could not (it relied on being invoked
with scripts/ as the cwd).
"""

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
DEFAULT_ROOT = str(SCRIPTS / "local")
