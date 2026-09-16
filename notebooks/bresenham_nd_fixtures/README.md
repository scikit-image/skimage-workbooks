# N-D Bresenham reference fixtures

Generated against `_bresenham_nd` in the `bresenham-nd` worktree.

## Regenerate

From the `notebooks` directory, with the `bresenham-nd` env and build on
the path:

```bash
export PYENV_VERSION=bresenham-nd
python bresenham_nd_fixtures/generate_fixtures.py \
  --skimage-root ../bresenham-nd/build-install/usr/lib/python3.13/site-packages
```

If that site-packages path is already on `PYTHONPATH`, `--skimage-root` can be
omitted (the script also searches `../bresenham-nd/build-install/...`).

Optional: pass `--zingl-bin /path/to/zingl_line3d` to assert the 3-D Zingl
field matches a compiled build of `zingl_line3d.c` (Alois Zingl's own
published `plotLine3d`, see below). The fixtures themselves store the Python
`zingl_line` sequences, which match that binary on the corpus.

`zingl_line3d.c` in this directory is Zingl's own published 3-D Bresenham
source, copied verbatim (see `/copyright` at the repo root — this file is
not covered by the repo's own license). Build it with
`make -C ../.. notebooks/bresenham_nd_fixtures/zingl_line3d` (or plain
`cc -O2 -o zingl_line3d zingl_line3d.c`) for a `--zingl-bin`-compatible CLI.
`bresenham_nd_cython.md` uses it for its 3-D cross-check.

Algorithms live in `references.py` (ITK Index→Index port, Zingl /
raster_geometry). The generator is `generate_fixtures.py`.

## Other implementations (not only ITK)

| Source | Dims | Notes |
| --- | --- | --- |
| **raster_geometry** `bresenham_line` | N-D | Pure Python. Pip package currently broken on NumPy 2 (`np.float_`); algorithm vendored in `references.py`. |
| **Zingl** `plotLine3d` | 3-D only | C source, vendored verbatim in `zingl_line3d.c` (see `/copyright`). No generic N-D API. Same rule as `raster_geometry`, since `raster_geometry` is itself modelled on it. |
| **ITK** `BresenhamLine` | N-D | Gold imaging reference. Python wheels do not wrap this class; fixtures use a port of `itkBresenhamLine.hxx` from ITK **v5.4.0**. Index→Index goes through a **normalised float direction**. |
| ActiveState recipe 578112 | N-D | Skipped: `nslope` + `np.rint` (DDA-style), not integer Bresenham. |
| encukou `bresenham` | 2-D only | Not N-D. |

So there **are** other N-D options: `raster_geometry` is the practical Python one. Zingl's own source covers 3-D only. ITK remains the main imaging-library reference.

## Agreement (same corpora as the JSON)

On a ±3 2-D box (2401 pairs), random 3-D/4-D samples (seed 0):

| | 2-D | 3-D | 4-D |
| --- | ---: | ---: | ---: |
| ours vs `skimage.draw.line` / `_line` | **100%** | — | — |
| ours vs ITK port | 71% | 77% | 64% |
| ours vs raster_geometry | 71% | 57% | 43% |
| ITK vs raster_geometry | 90% | 80% | 79% |
| raster_geometry vs Zingl plotLine3d | — | **100%** | — |

Disagreements are **tie-breaking**, not length: all keep Chebyshev length `max(|Δ|) + 1`. Example `(0,0,0)→(2,4,8)`: ours steps the middle axes earlier than Zingl/raster; both sequences have 9 points.

`_bresenham_nd` is locked to scikit-image’s 2-D Bresenham, not to ITK.

## Files

- `nd_bresenham_references.json` — all refs side by side (`ours`, `skimage_line`, `itk`, `raster_geometry`, `zingl_line3d` where applicable) plus agreement table.
- `itk_bresenham_line.json` — ITK port pixels only (external imaging reference).
- `ours_bresenham_nd.json` — `_bresenham_nd` pixels (regression oracle for this worktree).
- `references.py` — pure-Python ITK / Zingl ports.
- `generate_fixtures.py` — regenerate the three JSON files.
