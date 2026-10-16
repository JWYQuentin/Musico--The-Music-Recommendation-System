import subprocess
import sys


def test_the_service_loads_no_torch_faiss_or_implicit():
    """The API's process must not hold PyTorch or FAISS next to XGBoost, and its image installs none of the three."""
    code = "import sys, m4a_rec.api; print(sorted({'torch', 'faiss', 'implicit'} & set(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
