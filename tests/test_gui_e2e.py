"""Headless GUI boot test; catches import and QApplication initialization regressions."""
import os
import subprocess
import sys
from pathlib import Path


def test_gui_starts_headless():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.setdefault("HF_TOKEN", "gui-test-does-not-use-token")
    result = subprocess.run(
        [sys.executable, str(root / "gui.py"), "--self-test"],
        cwd=root, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "GUI self-test OK" in result.stdout
