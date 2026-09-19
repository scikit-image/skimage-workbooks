# Hybrid Hessian Filter — Fig. 2 reference panels

Five panels from Fig. 2 of the Hybrid Hessian Filter paper, used by
`on_hessian_filter.md` §7 to replicate the paper's pipeline stage by stage.

## Copyright

These images are **not** covered by this repository's licence. See the
`Files: notebooks/hhf_fig2_fixtures/fig2*.png` stanza in `/LICENSE`.

> Ng, C.-C., Yap, M. H., Costen, N., Li, B.: Automatic Wrinkle Detection using
> Hybrid Hessian Filter. In: ACCV 2014, LNCS 9005, pp. 609–622. Springer
> International Publishing. <https://doi.org/10.1007/978-3-319-16811-1_40>

© Springer International Publishing. The authors' accepted manuscript, from
which these panels are taken, is distributed by Loughborough University's
institutional repository under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

They are reproduced here unaltered apart from conversion to 8-bit greyscale,
for the non-commercial purpose of verifying the algorithm the notebook
assesses, with attribution as above.

Only the five panels the comparison needs are committed. Panel (a), the colour
original, is deliberately omitted: the notebook does not use it. The underlying
forehead photograph is from the Bosphorus face database, which is not
redistributable and is **not** redistributed here — only the printed figure.

## Files

| file | paper's panel | what it is |
| --- | --- | --- |
| `fig2b_greyscale.png` | (b) | the greyscale forehead: the pipeline's input |
| `fig2c_gradient.png` | (c) | the directional gradient of eq. (1) |
| `fig2d_vesselness.png` | (d) | the Frangi filter, eqs. (2)–(15) |
| `fig2e_mask.png` | (e) | "ridge likeliness", the binary mask of eq. (16) |
| `fig2f_threshold.png` | (f) | after the 250 px area threshold |

All are 845×117, 8-bit greyscale, as printed at 365 ppi.

## Regenerate

Needs poppler's `pdfimages` and a local copy of the paper under the gitignored
`library/` symlink. Nothing in the site build needs either — the committed PNGs
are what the notebook reads.

```bash
make fixtures SET=hhf_fig2
# or, equivalently, from the repository root:
python library/generate_fixtures.py hhf_fig2
```

The generator lives in `library/` because that is where the papers are, and
because it is useless without them; it is not part of the repository. It
asserts each panel's shape, so it fails loudly if a different printing of the
PDF orders its embedded images differently.

## Verify

To check the committed panels against a fresh extraction, without writing
anything:

```bash
make check-fixtures
# or: python library/generate_fixtures.py --check
```

It reports, per panel, whether the pixels match and whether the file is
byte-identical, and exits non-zero on any pixel difference. Extraction is
deterministic: the committed files reproduce byte-for-byte, so a mismatch
means either the fixtures or the local copy of the paper has changed.
