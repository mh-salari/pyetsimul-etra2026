"""Fetch the title-slide badges as SVG, exactly as they render online.

shields.io (and the pepy / readthedocs endpoints) already return SVG, so we just
download each one and save it. This gives the exact original look -- glossy
gradient, fonts, spacing, live values -- as a static local file that scales
crisply at any size (PowerPoint 2016+ imports SVG natively). Re-run to refresh
the live numbers.

Output: ../img/badges/badge_*.svg. Run:
    uv run python presentation/scripts/generate_badges.py
"""

import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "img" / "badges"

# (output filename, badge URL) -- the same endpoints the title slide references,
# so the saved SVGs match what GitHub/the live deck render.
BADGES: list[tuple[str, str]] = [
    ("badge_pypi.svg", "https://img.shields.io/pypi/v/pyetsimul"),
    ("badge_downloads.svg", "https://static.pepy.tech/badge/pyetsimul"),
    ("badge_license.svg", "https://img.shields.io/badge/License-GPL--3.0-green"),
    ("badge_docs.svg", "https://readthedocs.org/projects/pyetsimul/badge/?version=latest"),
    ("badge_doi.svg", "https://img.shields.io/badge/DOI-10.1145%2F3806023-blue"),
]


def fetch_svg(url: str) -> str:
    """Download a badge endpoint and return its SVG markup."""
    req = urllib.request.Request(url, headers={"User-Agent": "badge-generator"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8", errors="replace")
    if "svg" not in content_type and "<svg" not in body:
        raise ValueError(f"{url} did not return SVG (Content-Type: {content_type})")
    return body


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in BADGES:
        (OUT_DIR / filename).write_text(fetch_svg(url))
        print(f"wrote {filename}  <- {url}")


if __name__ == "__main__":
    main()
