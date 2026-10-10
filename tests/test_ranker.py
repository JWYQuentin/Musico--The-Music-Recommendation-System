"""XGBoost crashes when loaded into a process that has loaded PyTorch, as this test run has.

So the ranker's tests live in tests/isolated/, which the normal run skips, and run here in
a process of their own. This file must not import m4a_rec.ranker or xgboost.
"""
import subprocess
import sys
from pathlib import Path


def test_ranker_tests_pass_in_their_own_process():
    isolated = Path(__file__).parent / "isolated"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(isolated)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert " passed" in done.stdout and "no tests ran" not in done.stdout
