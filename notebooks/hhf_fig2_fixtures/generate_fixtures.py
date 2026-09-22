#!/usr/bin/env python3
"""Extract the Fig. 2 reference panels from the Hybrid Hessian Filter paper.

The five panels are committed next to this script; the notebook reads those.
Run this only to regenerate them, or with --check to verify them against a
fresh extraction. Needs poppler's `pdfimages`.

The source is the authors' accepted manuscript, distributed by Loughborough
University under CC BY-NC-ND 4.0. See README.md for the citation and the
copyright terms, which are not this repository's own:

    https://repository.lboro.ac.uk/articles/journal_contribution/\
Automatic_wrinkle_detection_using_hybrid_hessian_filter/9403112
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# Fig. 2 is on page 5 of the accepted manuscript. Its panels are the only
# images of this size on the page, and they appear in panel order. Panel (a),
# the colour original, is 845x116 and is deliberately not extracted.
PAGE = 5
PANEL_SHAPE = (117, 845)
PANELS = [
    "fig2b_greyscale.png",
    "fig2c_gradient.png",
    "fig2d_vesselness.png",
    "fig2e_mask.png",
    "fig2f_threshold.png",
]


def extract(pdf):
    """Return the Fig. 2 panels from `pdf`, in panel order, as 8-bit arrays."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["pdfimages", "-f", str(PAGE), "-l", str(PAGE), "-png",
                 str(pdf), f"{tmp}/p"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise SystemExit("pdfimages not found; install poppler-utils")
        except subprocess.CalledProcessError as err:
            raise SystemExit(
                f"pdfimages could not read {pdf}: "
                f"{err.stderr.decode().strip()}"
            )
        panels = []
        for path in sorted(Path(tmp).glob("p-*.png")):
            image = Image.open(path)
            if image.size == PANEL_SHAPE[::-1]:
                panels.append(np.array(image.convert("L")))
    if len(panels) != len(PANELS):
        raise SystemExit(
            f"expected {len(PANELS)} panels of {PANEL_SHAPE[::-1]} on page "
            f"{PAGE}, found {len(panels)}; is this a different printing?"
        )
    return panels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path, help="the accepted manuscript")
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed panels instead of writing them",
    )
    args = ap.parse_args(argv)

    panels = extract(args.pdf)

    if not args.check:
        for name, panel in zip(PANELS, panels):
            Image.fromarray(panel).save(HERE / name)
        return 0

    failed = False
    for name, panel in zip(PANELS, panels):
        committed = np.array(Image.open(HERE / name))
        same_pixels = committed.shape == panel.shape and (committed == panel).all()
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.fromarray(panel).save(tmp.name)
            same_bytes = Path(tmp.name).read_bytes() == (HERE / name).read_bytes()
        print(f"{name}: pixels {'match' if same_pixels else 'DIFFER'}, "
              f"bytes {'identical' if same_bytes else 'differ'}")
        failed |= not same_pixels
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
