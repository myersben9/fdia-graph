"""Render a Mermaid source file to a standalone SVG and a PNG next to it. Every diagram in the docs
is such a rendered image, because the README is also the PyPI page and the docs are read in the
GitHub mobile app, and neither renders a Mermaid block.

    python tools/render_mermaid.py docs/figures/diagrams/<name>.mmd [more.mmd ...]

A diagram is rendered one of two ways, chosen by its file name (`is_generated`):

- a hand-written flow diagram is drawn for a phone: 20 px type, the site's teal palette, the diagram
  at its natural size (never scaled to the page) on a white card with padding and rounded corners, so
  the same image reads on a dark page and on a light one. Its PNG is two device pixels per CSS pixel
  and at most `MAX_PNG_WIDTH` wide, which `tools/prereview.py` checks with the other rules of
  `flow_faults`: top to bottom, at most `MAX_NODES` nodes.
- a generated class or module diagram (`classes_*`, `modules`, written by `tools/class_diagrams.py`)
  cannot fit a phone and keeps the wide rendering: Mermaid's default theme, scaled to a 1400 px page.

Labels are plain SVG text (no HTML labels), so the file displays as an image everywhere. Needs
Playwright with its Chromium (`pip install playwright && playwright install chromium`) and network
access to fetch the pinned mermaid.min.js from jsDelivr once per run; bump the version in MERMAID
deliberately, since a new Mermaid can change the layout of every checked-in render. Commit the
.mmd, the .png and the .svg together.
"""

from __future__ import annotations

import html
import json
import os
import re
import struct
import sys
import tempfile

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js"

# the diagrams tools/class_diagrams.py writes: classes_<group> and modules
GENERATED_PREFIX, GENERATED_MODULES = "classes_", "modules"
MAX_PNG_WIDTH = 900  # device pixels: 450 CSS pixels, which a 390 px phone shows at 15 px type or more
MAX_NODES = 8
SCALE = 2  # device pixels per CSS pixel of a PNG
CARD = "#ffffff"
CARD_PADDING = 16  # CSS pixels of card around the diagram
CARD_RADIUS = 12

_COMMON = {
    "startOnLoad": True,
    "securityLevel": "loose",
    "htmlLabels": False,
    "deterministicIds": True,
    "deterministicIDSeed": "fdia-graph",
}
WIDE = {**_COMMON, "flowchart": {"htmlLabels": False}}
PHONE = {
    **_COMMON,
    "theme": "base",
    "themeVariables": {
        "fontSize": "20px",
        "primaryColor": "#e0f2f1",
        "primaryBorderColor": "#00695c",
        "primaryTextColor": "#1b1f23",
        "lineColor": "#455a64",
        "textColor": "#1b1f23",
        "clusterBkg": "#f5fbfa",
        "clusterBorder": "#80cbc4",
        "edgeLabelBackground": CARD,
    },
    # Mermaid draws an edge label's background half transparent, so the arrow shows through the words
    "themeCSS": ".edgeLabel rect, .edgeLabel .background { opacity: 1 !important; fill: #ffffff !important; }",
    "flowchart": {
        "htmlLabels": False,
        "useMaxWidth": False,
        "wrappingWidth": 390,
        "nodeSpacing": 24,
        "rankSpacing": 36,
        "padding": 12,
    },
}

PAGE = """<!doctype html><meta charset="utf-8"><script src="{lib}"></script>
<body style="margin:0;background:{body}"><div id="card" style="{card}"><pre class="mermaid" style="margin:0">{code}</pre></div>
<script>mermaid.initialize({config});</script></body>"""
# every card is as wide as the widest allowed, so every diagram shows its type at the same size
CARD_STYLE = (
    f"display:inline-block;box-sizing:border-box;min-width:{MAX_PNG_WIDTH // SCALE}px;text-align:center;"
    f"background:{CARD};padding:{CARD_PADDING}px;border-radius:{CARD_RADIUS}px"
)
# the card behind a standalone SVG: a rectangle over the whole view box, under everything else
BACKDROP = """(() => {
  const svg = document.querySelector('.mermaid svg');
  const [x, y, w, h] = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  for (const [k, v] of Object.entries({x, y, width: w, height: h, rx: RADIUS, fill: 'CARD'}))
    rect.setAttribute(k, v);
  svg.insertBefore(rect, svg.firstChild);
})()"""


def is_generated(path: str) -> bool:
    """Whether the diagram is one `tools/class_diagrams.py` writes: those keep the wide rendering."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.startswith(GENERATED_PREFIX) or name == GENERATED_MODULES


def png_width(path: str) -> int:
    """The width of a PNG in pixels, read from its header."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">I", head[16:20])[0]


# The house style of a node: a name, then a quoted label in a box, a cylinder or a rounded box.
_STYLED = re.compile(r'(\w+)\s*(?:\[\(""\)\]|\[""\]|\(""\))')
_SUBGRAPH = re.compile(r"^\s*subgraph\s+(\w+)", re.M)
_NOT_NODES = ("subgraph", "end", "direction", "%%", "classDef", "class ", "style", "linkStyle")


def flow_nodes(code: str) -> tuple[set[str], set[str]]:
    """The nodes of a flow diagram's source, and those of them not written in the house style
    (`name["label"]`, `name[("label")]` or `name("label")`): a bare name in an arrow chain, or an
    unquoted label, whose words read as names here."""
    bare = re.sub(r'"[^"]*"', '""', code)  # a label's text names nothing, over however many lines
    boxes = set(_SUBGRAPH.findall(bare))  # a subgraph may end an arrow; it is no node
    styled = set(_STYLED.findall(bare)) - boxes
    named: set[str] = set()
    for line in bare.splitlines()[1:]:
        if not line.strip().startswith(_NOT_NODES):
            named.update(re.findall(r"\w+", _STYLED.sub(r" \1 ", line)))
    named -= boxes
    return named | styled, named - styled


def source_faults(code: str) -> list[str]:
    """The faults of a flow diagram that its source shows: not top to bottom, a node outside the
    house style, more than `MAX_NODES` nodes."""
    faults = []
    if not re.match(r"\s*flowchart\s+(TB|TD)\b", code):
        faults.append("is not `flowchart TB`")
    nodes, unstyled = flow_nodes(code)
    if unstyled:
        faults.append('has nodes not written as name["label"]: ' + ", ".join(sorted(unstyled)))
    if len(nodes) > MAX_NODES:
        faults.append(f"has {len(nodes)} nodes (at most {MAX_NODES})")
    return faults


def flow_faults(src: str) -> list[str]:
    """What keeps a hand-written flow diagram from reading on a phone: `source_faults`, or a PNG wider
    than `MAX_PNG_WIDTH`. Empty when it reads."""
    with open(src, encoding="utf8") as fh:
        faults = source_faults(fh.read())
    png = os.path.splitext(src)[0] + ".png"
    if os.path.exists(png) and png_width(png) > MAX_PNG_WIDTH:
        faults.append(f"its PNG is {png_width(png)} px wide (at most {MAX_PNG_WIDTH})")
    return faults


def render(src: str) -> tuple[str, str]:
    from playwright.sync_api import sync_playwright

    with open(src, encoding="utf8") as fh:
        code = fh.read()
    wide = is_generated(src)
    svg_path, png_path = os.path.splitext(src)[0] + ".svg", os.path.splitext(src)[0] + ".png"
    page_html = PAGE.format(
        lib=MERMAID,
        code=html.escape(code),
        body="#fff" if wide else "transparent",
        card="" if wide else CARD_STYLE,
        config=json.dumps(WIDE if wide else PHONE),
    )
    # a private directory per invocation: concurrent renders never share the page file, and nothing
    # pre-existing in the system temp directory can be followed or clobbered
    with tempfile.TemporaryDirectory(prefix="render_mermaid_") as tmp, sync_playwright() as p:
        page_path = os.path.join(tmp, "page.html")
        with open(page_path, "w", encoding="utf8") as fh:
            fh.write(page_html)
        browser = p.chromium.launch()
        viewport = {"width": 1400 if wide else 440, "height": 900}
        page = browser.new_page(viewport=viewport, device_scale_factor=SCALE)
        page.goto("file:///" + page_path.replace("\\", "/"))
        page.wait_for_function("document.querySelector('.mermaid svg')", timeout=60000)
        if wide:
            page.locator(".mermaid svg").screenshot(path=png_path)
        else:  # the card, its corners left transparent
            page.locator("#card").screenshot(path=png_path, omit_background=True)
            page.evaluate(BACKDROP.replace("RADIUS", str(CARD_RADIUS)).replace("CARD", CARD))
        # XMLSerializer, not outerHTML: an HTML serialization leaves void tags unclosed and the file is
        # then not well-formed XML, which image viewers refuse
        svg = page.evaluate("new XMLSerializer().serializeToString(document.querySelector('.mermaid svg'))")
        browser.close()
    if 'aria-roledescription="error"' in svg:  # Mermaid draws its error picture instead of the diagram
        raise SystemExit(f"{src}: Mermaid could not parse the source; the images written show its error")
    if "<foreignObject" in svg:  # an HTML label slipped through: the file would show boxes without text
        raise SystemExit(
            f"{src}: the SVG still carries HTML labels (foreignObject); check the Mermaid config"
        )
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    with open(svg_path, "w", encoding="utf8", newline="\n") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n' + svg)
    return svg_path, png_path


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    failed = False
    for src in argv:
        svg_path, png_path = render(src)
        print(f"{src} -> {os.path.basename(svg_path)}, {os.path.basename(png_path)}")
        for fault in [] if is_generated(src) else flow_faults(src):
            failed = True
            print(f"  {src} {fault}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
