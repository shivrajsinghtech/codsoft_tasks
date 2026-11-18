import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "predict.py"), *args], capture_output=True,
                          text=True, cwd=ROOT, timeout=120)


def test_cli_single_prediction(tiny_model_path):
    r = run("--model-path", str(tiny_model_path), "--plot", "A ghost haunts a graveyard with a demon at night")
    assert r.returncode == 0, r.stderr
    assert "Predicted genre: horror" in r.stdout and "Top 3" in r.stdout


def test_cli_empty_plot(tiny_model_path):
    r = run("--model-path", str(tiny_model_path), "--plot", "   ")
    assert r.returncode == 1 and "Invalid input" in r.stderr


def test_cli_batch(tiny_model_path, tiny_data_dir, tmp_path):
    out = tmp_path / "pred.csv"
    r = run("--model-path", str(tiny_model_path), "--input", str(tiny_data_dir / "test_data.txt"),
            "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert len(out.read_text().strip().splitlines()) == 13


def test_cli_missing_model(tmp_path):
    r = run("--model-path", str(tmp_path / "none.joblib"), "--plot", "a b c d")
    assert r.returncode == 2
