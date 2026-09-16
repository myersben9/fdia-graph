"""The models section of the data dictionary is generated from the dataclasses; this keeps it so."""

import os
import subprocess
import sys

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "models_doc.py")


def test_models_section_is_current():
    r = subprocess.run([sys.executable, TOOL, "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
