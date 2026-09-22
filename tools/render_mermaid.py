"""Render a Mermaid source file to a standalone SVG (and a PNG next to it), for the diagrams that
have to show where Mermaid is not rendered: the README on PyPI and in the GitHub mobile app.

    python tools/render_mermaid.py docs/figures/pipeline.mmd [more.mmd ...]

Labels are plain SVG text (no HTML labels), so the file displays as an image everywhere. Needs
Playwright with its Chromium (`pip install playwright && playwright install chromium`) and network
access to fetch the pinned mermaid.min.js from jsDelivr once per run; bump the version in MERMAID
deliberately, since a new Mermaid can change the layout of every checked-in render. The other diagrams in docs/ stay as
Mermaid blocks, which github.com renders in place.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js"

PAGE = """<!doctype html><meta charset="utf-8"><script src="{lib}"></script>
<body style="margin:0;background:#fff"><pre class="mermaid">{code}</pre>
<script>mermaid.initialize({{startOnLoad: true, securityLevel: "loose", htmlLabels: false, flowchart: {{htmlLabels: false}}, deterministicIds: true, deterministicIDSeed: "fdia-graph"}});</script></body>"""


def render(src: str) -> tuple[str, str]:
    from playwright.sync_api import sync_playwright

    with open(src, encoding="utf8") as fh:
        code = fh.read()
    svg_path, png_path = os.path.splitext(src)[0] + ".svg", os.path.splitext(src)[0] + ".png"
    # a private directory per invocation: concurrent renders never share the page file, and nothing
    # pre-existing in the system temp directory can be followed or clobbered
    with tempfile.TemporaryDirectory(prefix="render_mermaid_") as tmp, sync_playwright() as p:
        page_path = os.path.join(tmp, "page.html")
        with open(page_path, "w", encoding="utf8") as fh:
            fh.write(PAGE.format(lib=MERMAID, code=html.escape(code)))
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        page.goto("file:///" + page_path.replace("\\", "/"))
        page.wait_for_function("document.querySelector('.mermaid svg')", timeout=60000)
        # XMLSerializer, not outerHTML: an HTML serialization leaves void tags unclosed and the file is
        # then not well-formed XML, which image viewers refuse
        svg = page.evaluate("new XMLSerializer().serializeToString(document.querySelector('.mermaid svg'))")
        page.locator(".mermaid svg").screenshot(path=png_path)
        browser.close()
    if "<foreignObject" in svg:  # an HTML label slipped through: the file would show boxes without text
        raise SystemExit(
            f"{src}: the SVG still carries HTML labels (foreignObject); check the Mermaid config"
        )
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    with open(svg_path, "w", encoding="utf8", newline="\n") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n' + svg)
    return svg_path, png_path


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    for src in argv:
        svg_path, png_path = render(src)
        print(f"{src} -> {os.path.basename(svg_path)}, {os.path.basename(png_path)}")


if __name__ == "__main__":
    main(sys.argv[1:])
