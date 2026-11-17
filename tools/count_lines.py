"""Count the lines of the two toolkits: all lines, code lines (no docstring, comment or blank line),
and the code inside and outside the homes of constants (`constants/` and `schema.py`).

    python tools/count_lines.py [root]    # root: another checkout to count (default: this one)
"""

import ast
import io
import sys
import tokenize
from pathlib import Path

from sources import PACKAGES, ROOT


class LineCount:
    """The line counts of each toolkit package of a checkout."""

    HOMES = ("constants/", "schema.py")  # where constants live, relative to a package
    BLANK = ("COMMENT", "NL", "NEWLINE", "INDENT", "DEDENT", "ENDMARKER")  # tokens that are not code
    DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

    @classmethod
    def code_lines(cls, text: str) -> int:
        """The lines of `text` that hold code."""
        doc: set[int] = set()
        for node in ast.walk(ast.parse(text)):
            first = node.body[0] if isinstance(node, cls.DOCUMENTED) and node.body else None
            if isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str):
                doc.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
        code: set[int] = set()
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tokenize.tok_name[tok.type] not in cls.BLANK:
                code.update(range(tok.start[0], tok.end[0] + 1))
        return len(code - doc)

    @classmethod
    def package(cls, folder: Path) -> tuple[int, int, int]:
        """(all lines, code lines, code lines in the homes of constants) of the package at `folder`."""
        total = code = home = 0
        for path in sorted(folder.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            lines = cls.code_lines(text)
            total, code = total + text.count("\n"), code + lines
            home += lines if path.relative_to(folder).as_posix().startswith(cls.HOMES) else 0
        return total, code, home

    @classmethod
    def main(cls, root: Path) -> int:
        print(f"{'package':15s}{'lines':>8s}{'code':>8s}{'constants':>11s}{'other code':>12s}")
        for name, folder in PACKAGES.items():
            total, code, home = cls.package(root / folder)
            print(f"{name:15s}{total:8d}{code:8d}{home:11d}{code - home:12d}")
        return 0


if __name__ == "__main__":
    sys.exit(LineCount.main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(ROOT)))
