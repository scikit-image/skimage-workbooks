"""Shared helpers for MyST / Jupyter workbooks."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import pandas as pd
from IPython.display import Markdown, display

IndexArg = bool | str | Hashable | Sequence[Hashable] | None

# ASCII | is the Markdown table delimiter; ∣ (U+2223 DIVIDES) looks the same.
_MD_PIPE = "|"
_MD_PIPE_GLYPH = "\u2223"


def _markdown_safe_value(value):
    """Replace Markdown table separators in labels/cells; leave other values."""
    if isinstance(value, str):
        return value.replace(_MD_PIPE, _MD_PIPE_GLYPH)
    if isinstance(value, tuple):
        return tuple(_markdown_safe_value(v) for v in value)
    return value


def _markdown_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame so to_markdown cannot mis-parse | in headers or cells."""
    out = df.copy()
    out.columns = [_markdown_safe_value(c) for c in out.columns]
    if isinstance(out.index, pd.MultiIndex):
        out.index = pd.MultiIndex.from_tuples(
            [_markdown_safe_value(t) for t in out.index],
            names=[_markdown_safe_value(n) for n in out.index.names],
        )
    else:
        name = out.index.name
        out.index = out.index.map(_markdown_safe_value)
        out.index.name = _markdown_safe_value(name)
    return out.map(_markdown_safe_value)


def show_table(df: pd.DataFrame, index: IndexArg = None, **kwargs):
    """Render a DataFrame as a Markdown table (keeps formatting in HTML and PDF).

    Bare DataFrame outputs use ``text/html`` / ``text/plain``; PDF builders fall
    back to the monospace plain-text repr. Markdown tables survive both.

    ``index`` selects the row index and whether to print it:

    - ``None`` (default): show the index only when it is meaningful (named, or
      not a plain ``RangeIndex``).
    - ``True`` / ``False``: force showing or hiding the current index.
    - column label or list of labels: ``set_index`` those columns, then show.

    ASCII ``|`` in headers or cells is rewritten to U+2223 (∣) so Markdown
    table parsers do not treat it as a column separator.

    Extra ``kwargs`` go to ``DataFrame.to_markdown`` (e.g. ``floatfmt``).
    """
    if index is not None and not isinstance(index, bool):
        df = df.set_index(index)
        index = True
    elif index is None:
        index = df.index.name is not None or not isinstance(df.index, pd.RangeIndex)
    display(Markdown(_markdown_safe_frame(df).to_markdown(index=index, **kwargs)))
