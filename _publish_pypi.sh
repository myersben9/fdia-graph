#!/usr/bin/env bash
# Manual fallback: publish the CURRENT pyproject version of fdia-graph to PyPI.
# (The normal path is automatic: push a vX.Y.Z tag and .github/workflows/publish.yml
#  builds + publishes via PyPI Trusted Publishing, no token anywhere.)
#
# USAGE:
#   TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXX ./_publish_pypi.sh
#   PYPI_REPO=testpypi ./_publish_pypi.sh     # dry-run against TestPyPI
#   PYTHON=/path/to/python ./_publish_pypi.sh # override interpreter
set -euo pipefail
cd "$(dirname "$0")"

# Interpreter: $PYTHON if set, else the repo venv (either layout), else python3/python from PATH.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "../venv/python.exe" ]; then PY="../venv/python.exe"     # Windows venv layout
elif [ -x "../venv/bin/python" ]; then PY="../venv/bin/python"     # Unix venv layout
elif command -v python3 >/dev/null 2>&1; then PY="python3"
else PY="python"; fi
REPO="${PYPI_REPO:-pypi}"

# Version comes from pyproject.toml -- never hardcoded, so a bump can't be missed here.
VERSION="$("$PY" -c "import re;print(re.search(r'^version = \"([^\"]+)\"', open('pyproject.toml').read(), re.M).group(1))")"
echo "== publishing fdia-graph ${VERSION} to ${REPO} =="

rm -rf dist build src/*.egg-info
"$PY" -m build
"$PY" -m twine check dist/fdia_graph-"${VERSION}"*

if [ "$REPO" = "testpypi" ]; then
  "$PY" -m twine upload --non-interactive --repository testpypi dist/fdia_graph-"${VERSION}"*
else
  "$PY" -m twine upload --non-interactive dist/fdia_graph-"${VERSION}"*
fi
echo "== done. verify: pip index versions fdia-graph (or https://pypi.org/project/fdia-graph/${VERSION}/) =="
