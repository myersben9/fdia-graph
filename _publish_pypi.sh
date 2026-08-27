#!/usr/bin/env bash
# Publish fdia-graph 0.7.6 to PyPI. The dist/ artifacts are already built and `twine check`-clean.
#
# USAGE (Ben runs this — the token stays in your shell, never written to a file or committed):
#   1. Get a PyPI API token: https://pypi.org/manage/account/token/  (scope it to the fdia-graph project)
#   2. From the SDK dir, run ONE of:
#        TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXXXXXX ./_publish_pypi.sh
#      or just run it and paste the token when twine prompts:
#        ./_publish_pypi.sh
#
# To dry-run against TestPyPI first (recommended), set:  PYPI_REPO=testpypi ./_publish_pypi.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV="../venv/python.exe"
REPO="${PYPI_REPO:-pypi}"

echo "== re-checking artifacts =="
ls -la dist/
$VENV -m twine check dist/*

echo "== uploading fdia-graph 0.7.6 to ${REPO} =="
if [ "$REPO" = "testpypi" ]; then
  $VENV -m twine upload --repository testpypi dist/fdia_graph-0.7.6*
else
  $VENV -m twine upload dist/fdia_graph-0.7.6*
fi

echo "== done. verify with: pip install --upgrade fdia-graph  (should now report 0.7.6) =="
