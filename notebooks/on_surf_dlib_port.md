---
title: 'On the dlib port of the SURF determinant'
date: 2026-11-28
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{related}
https://github.com/davisking/dlib/blob/a7b38a1f12fda2e99f207d501ff9c8a5d040b90f/dlib/image_keypoint/hessian_pyramid.h | The dlib code that the new `_hessian_det_appx_pythran.py` ports; compared in [§4](#sec-port-calls) and [§6](#sec-port-names)
https://github.com/gussmith23/opensurf/blob/409d143f8d896f9a4b8d00d32d5880515d569953/src/fasthessian.cpp | The OpenSURF code that the old file translates; compared in [§4](#sec-port-calls) and [§6](#sec-port-names)
:::

:::{changelog}
2026-11-28 | claude-code:claude-opus-5-5 | Create workbook
:::

`hessian_matrix_det(..., approximate=True)` and `blob_doh` compute the SURF
determinant of the Hessian with box filters [@bay2008speededup_robust]. The
code is `_hessian_matrix_det` in `_hessian_det_appx_pythran.py`.
`on_hessian_det_approx.md` §9.3 shows that the current code on `main` is a
close translation of OpenSURF [@evans2009notes_opensurf], which is under the
GPL. The `surf-fixes` branch replaces it with a port of dlib [@king2009dlibml],
which is under the Boost Software License.

This notebook tests two claims about the new code:

1. it makes the same box-sum calls as dlib, with the same arguments, in the
   same order;
2. it keeps none of the forms by which the old code follows OpenSURF.

The method is to run all four codes. The two Python files run as written.
For the two C++ files, the notebook takes the statements that build the
filters from the source text and evaluates them in Python, with functions
that record each box-sum call.

Coordinates are in array order: $r$ runs down the rows, $c$ across the
columns. dlib's `point(x, y)` has $x = c$ and $y = r$.

## 1. Setup

```{code-cell} ipython3
import ast
import builtins
import hashlib
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pooch

from nbhelper import show_table
```

The subject under test is `hessian_det_appx_dlib_port.py`, beside this
notebook. It is a copy of `src/_skimage2/feature/_hessian_det_appx_pythran.py`
on the `surf-fixes` branch of scikit-image. The hash below identifies the
copy.

```{code-cell} ipython3
NEW_PATH = pathlib.Path("hessian_det_appx_dlib_port.py")
NEW_SHA256 = "9c151e192f48378494b432cd6ef26c27e14216469466ac05ce151c71147c6b58"
NEW_SOURCE = NEW_PATH.read_text()
assert hashlib.sha256(NEW_SOURCE.encode()).hexdigest() == NEW_SHA256
```

The comparators are fetched at fixed commits: the old file from scikit-image
`main`, OpenSURF from `gussmith23/opensurf`, and dlib from `davisking/dlib`.
`pooch` checks each file against its SHA-256 hash.

```{code-cell} ipython3
# Comparators.
SKIMAGE = ("https://raw.githubusercontent.com/scikit-image/scikit-image/"
           "2b5601b68184d66b241eb6c94286918452f0cf3c/src/_skimage2/feature/")
OPENSURF = ("https://raw.githubusercontent.com/gussmith23/opensurf/"
            "409d143f8d896f9a4b8d00d32d5880515d569953/")
DLIB = ("https://raw.githubusercontent.com/davisking/dlib/"
        "a7b38a1f12fda2e99f207d501ff9c8a5d040b90f/dlib/")
FILES = {
    "old": (SKIMAGE + "_hessian_det_appx_pythran.py",
            "e149cefecb6694b356d9080f9c3851d6d81a4c4a7e786af479ddbf250c42bf59"),
    "fasthessian.cpp": (OPENSURF + "src/fasthessian.cpp",
                        "4a67afba0695037a8f34a0b3ac519cbd5839591c8d0690f1acc2f8e4dcab566c"),
    "integral.h": (OPENSURF + "include/integral.h",
                   "2763c97029f993db317cf7fc535c2cb85b78393f4f7cb97e0b74ea4169308e54"),
    "hessian_pyramid.h": (DLIB + "image_keypoint/hessian_pyramid.h",
                          "cb178cda84ac1f28593d649610e25a1a3900acc5d9292f7d30ed553df9b11550"),
    "integral_image.h": (DLIB + "image_transforms/integral_image.h",
                         "f3ddd3f69748a09329ea33d8b10398642d2e958041bbef4a358def0dfa46a30c"),
    "rectangle.h": (DLIB + "geometry/rectangle.h",
                    "3941093dc8080e28c2ab6082992a95a9c71c3ccb52ce87180746bf3e603279ce"),
}
TEXT = {name: pathlib.Path(pooch.retrieve(url, known_hash=f"sha256:{sha}")).read_text()
        for name, (url, sha) in FILES.items()}
```

mahotas runs dlib's code from a port of its own [@coelho2013mahotas]; §7
uses it as an executable check.

```{code-cell} ipython3
import mahotas
from mahotas.features import _surf as mahotas_surf
from mahotas.features.surf import integral as mahotas_integral

print("mahotas", mahotas.__version__)
```

```{code-cell} ipython3
# Categorical slots 1-2 of the default palette, as in
# on_hessian_det_approx.md. validate_palette.js, light mode: all checks
# pass; contrast WARN for slot 2, so the figure labels each panel.
COLOURS = {"centred": "#257de0", "top-left": "#f47230"}
INK, MUTED, GRID = "#191919", "#666666", "#dedcd5"
plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "axes.edgecolor": GRID,
     "figure.facecolor": "white"}
)
```

## 2. Helpers

### 2.1 Running the Python files

`load_module` runs a source text as a module. Both Python files are plain
Python: the `# pythran export` lines are comments.

```{code-cell} ipython3
def load_module(source, name):
    """Namespace of `source`, run as a module called `name`."""
    namespace = {"__name__": name}
    exec(compile(source, name, "exec"), namespace)
    return namespace


NEW = load_module(NEW_SOURCE, "new")
OLD = load_module(TEXT["old"], "old")
```

`record_calls` replaces one function of a module with a wrapper that records
its arguments, then computes the determinant for one pixel. On a 1 × 1
integral image the only pixel is $(0, 0)$, so the recorded rows and columns
are offsets from the output pixel.

```{code-cell} ipython3
def record_calls(module, box_function, lobe):
    """Arguments after `img` of each call to `box_function`, at one pixel."""
    calls = []
    original = module[box_function]

    def wrapper(img, *args):
        calls.append(tuple(int(a) for a in args))
        return original(img, *args)

    module[box_function] = wrapper
    try:
        module["_hessian_matrix_det"](np.zeros((1, 1)), lobe)
    finally:
        module[box_function] = original
    return calls
```

### 2.2 Running the C++ statements

`function_body` cuts out the brace block of a C++ function. `strip_comments`
removes `//` and `/* */` comments.

```{code-cell} ipython3
def function_body(text, signature):
    """Text of the brace block after the first match of regex `signature`."""
    start = text.index("{", re.search(signature, text).end())
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
    raise ValueError("unbalanced braces")


def strip_comments(text):
    """C++ text without comments."""
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


BODIES = {
    "OpenSURF": {
        "buildResponseLayer": function_body(
            TEXT["fasthessian.cpp"], r"void FastHessian::buildResponseLayer"),
        "BoxIntegral": function_body(TEXT["integral.h"], r"inline float BoxIntegral"),
    },
    "dlib": {
        "build_pyramid": function_body(TEXT["hessian_pyramid.h"], r"void build_pyramid \("),
        "get_sum_of_area": function_body(
            TEXT["integral_image.h"], r"value_type get_sum_of_area \("),
        "centered_rect": function_body(
            TEXT["rectangle.h"], r"inline const rectangle centered_rect \(\s*long x"),
    },
}
```

`statement` finds `name = …;` in a body and returns the right-hand side.
`as_python` makes it a Python expression. The C++ statements here use only
names, calls, `+`, `-`, `*`, `/`, `->` and `::`. Between the positive
integers of the filter geometry, C++ `/` is Python `//`; for the one
floating-point statement, `area_inv`, it is `/`. The expression is put in
brackets so that it can span lines.

```{code-cell} ipython3
def statement(body, name):
    """Right-hand side of the first C++ statement `... name = ...;`."""
    match = re.search(rf"\b{name}\s*=\s*(.*?);", strip_comments(body), re.S)
    return match.group(1)


def as_python(expression, integer=True):
    """A C++ arithmetic expression as Python source; `/` is `//` if `integer`."""
    expression = expression.replace("->", ".").replace("::", ".")
    if integer:
        expression = re.sub(r"(?<![/])/(?![/])", "//", expression)
    return "(" + expression + ")"
```

The functions that the statements call are defined from their own source.

* OpenSURF's `BoxIntegral(img, row, col, rows, cols)` names the top-left
  pixel of a box and its numbers of rows and columns. It reads the integral
  image at `row - 1` and `row + rows - 1`; its comment says "The subtraction
  by one for row/col is because row/col is inclusive". So it sums rows
  `row .. row + rows - 1`.
* dlib's `centered_rect(p, width, height)` names the centre of a box and its
  width and height. Its source, printed below, sets `left = x - width / 2`,
  `top = y - height / 2`, and an inclusive right and bottom.

```{code-cell} ipython3
print(strip_comments(BODIES["dlib"]["centered_rect"]).strip())
```

```{code-cell} ipython3
class Point:
    """dlib `point(x, y)`: `x` is the column, `y` the row."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)


def dlib_centered_rect(p, width, height):
    """dlib `centered_rect`, as `(top, left, bottom, right)`, inclusive."""
    left, top = p.x - width // 2, p.y - height // 2
    return top, left, top + height - 1, left + width - 1
```

`opensurf_calls` and `dlib_calls` evaluate the three statements of each
library for one pixel at $(0, 0)$ and record each box call.

```{code-cell} ipython3
def opensurf_calls(lobe):
    """`(row, col, rows, cols)` of each `BoxIntegral` call, at one pixel."""
    body = BODIES["OpenSURF"]["buildResponseLayer"]
    calls = []

    def BoxIntegral(img, row, col, rows, cols):
        calls.append((row, col, rows, cols))
        return 0.0

    class Layer:
        filter = 3 * lobe

    env = {"BoxIntegral": BoxIntegral, "img": None, "r": 0, "c": 0, "rl": Layer}
    for name in ("b", "l", "w"):  # filter geometry, from the source
        env[name] = eval(as_python(statement(body, name)), env)
    for name in ("Dxx", "Dyy", "Dxy"):
        eval(as_python(statement(body, name)), env)
    return calls


class DlibImage:
    """A dlib integral image whose `get_sum_of_area` calls `sum_box`."""

    def __init__(self, sum_box):
        self.sum_box = sum_box

    def get_sum_of_area(self, rect):
        return self.sum_box(rect)


def dlib_env(lobe, r, c, sum_box, calls=None):
    """Names for dlib's statements at pixel `(r, c)` and `lobe`."""
    body = BODIES["dlib"]["build_pyramid"]

    def centered_rect(p, width, height):
        if calls is not None:
            calls.append((p.y, p.x, height, width))
        return dlib_centered_rect(p, width, height)

    env = {"centered_rect": centered_rect, "img": DlibImage(sum_box),
           "lobe_size": lobe, "r": r, "c": c}
    env["lobe_offset"] = eval(as_python(statement(body, "lobe_offset")), env)
    for name, x, y in re.findall(r"const point (\w+)\(([^,]+),([^)]+)\);",
                                 strip_comments(body)):
        env[name] = Point(eval(as_python(x), env), eval(as_python(y), env))
    return env


def dlib_calls(lobe):
    """`(r, c, height, width)` of each `centered_rect` call, at one pixel."""
    calls = []
    env = dlib_env(lobe, 0, 0, lambda rect: 0.0, calls)
    for name in ("Dxx", "Dyy", "Dxy"):
        eval(as_python(statement(BODIES["dlib"]["build_pyramid"], name)), env)
    return calls
```

### 2.3 The pixels a box call sums

For the two Python files, `footprint` finds the pixels a box call sums by
running it. It puts a single 1 in each pixel of a zero image in turn, and
records whether the box sum of that image is 1. The result is the
rectangle `(top, left, bottom, right)` relative to the output pixel.

```{code-cell} ipython3
def footprint(box_sum, args, n=41):
    """Rectangle `(top, left, bottom, right)` that `box_sum(img, *args)` sums."""
    h = n // 2
    r, c, *extent = args
    hit = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            impulse = np.zeros((n, n))
            impulse[i, j] = 1
            hit[i, j] = box_sum(impulse.cumsum(0).cumsum(1), h + r, h + c, *extent) == 1
    rows, cols = np.nonzero(hit)
    rect = (rows.min() - h, cols.min() - h, rows.max() - h, cols.max() - h)
    assert hit.sum() == (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    return tuple(int(v) for v in rect)
```

`python_names` lists the names used in the code of a Python file, and
`cpp_names` those in a C++ body. Neither counts comments, strings or
docstrings.

```{code-cell} ipython3
PYTHON_COMMON = set(dir(builtins)) | {"np", "self"}
CPP_COMMON = {
    "const", "long", "int", "float", "double", "unsigned", "char", "bool",
    "for", "if", "else", "return", "inline", "void", "std", "static_cast",
    "T", "value_type", "rectangle", "point", "min", "max", "pow", "lround",
    "sizeof", "true", "false", "this",
}


def python_names(source):
    """Names in the code of `source`, without a leading underscore."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return {n.lstrip("_") for n in names} - PYTHON_COMMON


def cpp_names(*bodies):
    """Identifiers in C++ `bodies`, without comments or string literals."""
    text = re.sub(r'"(?:\\.|[^"\\])*"', "", strip_comments("\n".join(bodies)))
    return set(re.findall(r"[A-Za-z_]\w*", text)) - CPP_COMMON
```

## 3. The four codes

The statements that build the three filters, as each library writes them.
OpenSURF and dlib are printed from the fetched source. These are short
extracts, of the lines this notebook evaluates.

```{code-cell} ipython3
for library, body in (("OpenSURF", BODIES["OpenSURF"]["buildResponseLayer"]),
                      ("dlib", BODIES["dlib"]["build_pyramid"])):
    print(f"--- {library}")
    for name in ("Dxx", "Dyy", "Dxy"):
        print(f"{name} = " + re.sub(r"\s+", " ", statement(body, name)))
```

The old and new Python code, from the function `_hessian_matrix_det` of each
file:

```{code-cell} ipython3
def python_statements(source, function):
    """Assignments in `function` of `source`, one line each."""
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == function)
    loop = [n for n in ast.walk(func) if isinstance(n, (ast.Assign, ast.AugAssign))]
    return [re.sub(r"\s+", " ", ast.unparse(n)) for n in loop]


for name, source in (("old", TEXT["old"]), ("new", NEW_SOURCE)):
    print(f"--- {name}")
    for line in python_statements(source, "_hessian_matrix_det"):
        if "_integ" in line or "_centered_sum" in line:
            print(line)
```

Each OpenSURF and old call names a box by a corner, written as an offset
from the output pixel, such as `r - l + 1` and `c - b`. Each dlib and new
call names a box by its centre, such as `p + bl` or
`r + lobe_offset, c - lobe_offset`, and its two sides.

(sec-port-calls)=
## 4. The box calls, side by side

The old code computes the boxes in a different order from OpenSURF: `Dxy`
first, and within `Dxy` in the order top-left, bottom-right, top-right,
bottom-left. `in_order_of` puts its calls in OpenSURF's order, so that each
row of the table below is one box.

```{code-cell} ipython3
def in_order_of(calls, reference):
    """Indices that put `calls` in the order of the same calls in `reference`."""
    return [calls.index(call) for call in reference]


LOBE = 3
CALLS = {
    "dlib": dlib_calls(LOBE),
    "new": record_calls(NEW, "_centered_sum", LOBE),
    "OpenSURF": opensurf_calls(LOBE),
    "old": record_calls(OLD, "_integ", LOBE),
}
OLD_ORDER = in_order_of(CALLS["old"], CALLS["OpenSURF"])
BOX_NAMES = ["Dxx outer", "Dxx middle", "Dyy outer", "Dyy middle",
             "Dxy 1", "Dxy 2", "Dxy 3", "Dxy 4"]
```

The recorded calls, at one pixel, for lobe 3. dlib's calls are
`(r, c, height, width)` of the centre and the box; the new code's are its
arguments `(r, c, height, width)`. OpenSURF's are `(row, col, rows, cols)`;
the old code's are its arguments `(r, c, rl, cl)`, in OpenSURF's order.

```{code-cell} ipython3
show_table(pd.DataFrame({
    "box": BOX_NAMES, "dlib": CALLS["dlib"], "new": CALLS["new"],
    "OpenSURF": CALLS["OpenSURF"],
    "old, in OpenSURF's order": [CALLS["old"][i] for i in OLD_ORDER],
}))
```

Each row is one call, in the order of each library. The `Dxy` rows of dlib
and OpenSURF are in different orders: dlib's first `Dxy` box is the
bottom-left one, OpenSURF's the top-right one.

The same check at more lobes. `same_calls` asks for the same calls in the
same order; `matched_calls` for the same calls in any order.

```{code-cell} ipython3
def same_calls(a, b):
    """True if `a` and `b` are the same calls, in the same order."""
    return a == b


def matched_calls(a, b):
    """True if `a` and `b` are the same calls, in any order."""
    return sorted(a) == sorted(b)


rows = []
for lobe in (1, 3, 5, 7, 9, 13):
    calls = {"dlib": dlib_calls(lobe), "new": record_calls(NEW, "_centered_sum", lobe),
             "OpenSURF": opensurf_calls(lobe), "old": record_calls(OLD, "_integ", lobe)}
    rows.append({
        "lobe": lobe,
        "new = dlib, in order": same_calls(calls["new"], calls["dlib"]),
        "new = OpenSURF, any order": matched_calls(calls["new"], calls["OpenSURF"]),
        "old = OpenSURF, any order": matched_calls(calls["old"], calls["OpenSURF"]),
        "old = dlib, any order": matched_calls(calls["old"], calls["dlib"]),
    })
show_table(pd.DataFrame(rows))
```

At every lobe the new code makes dlib's calls, with the same arguments, in
the same order, and its calls are not OpenSURF's. The old code makes
OpenSURF's calls, with the same arguments, at every lobe, and its calls are
not dlib's.

## 5. The pixels each call sums

The two conventions name the same boxes in different ways. `RECTS` gives
the rectangle each call sums, relative to the output pixel: from
`dlib_centered_rect` for dlib, from OpenSURF's inclusive rule for OpenSURF,
and from `footprint` for the two Python files, which runs their own box
sums.

```{code-cell} ipython3
def opensurf_rect(row, col, rows, cols):
    """The pixels `BoxIntegral(img, row, col, rows, cols)` sums."""
    return row, col, row + rows - 1, col + cols - 1


def rects(lobe):
    """Rectangle of each box call, for each of the four codes.

    The old code's rectangles are in OpenSURF's order.
    """
    old_calls = record_calls(OLD, "_integ", lobe)
    old_calls = [old_calls[i] for i in in_order_of(old_calls, opensurf_calls(lobe))]
    return {
        "dlib": [dlib_centered_rect(Point(c, r), w, h) for r, c, h, w in dlib_calls(lobe)],
        "new": [footprint(NEW["_centered_sum"], call)
                for call in record_calls(NEW, "_centered_sum", lobe)],
        "OpenSURF": [opensurf_rect(*call) for call in opensurf_calls(lobe)],
        "old": [footprint(OLD["_integ"], call) for call in old_calls],
    }


RECTS = rects(LOBE)
show_table(pd.DataFrame({"box": BOX_NAMES} | RECTS).rename(
    columns={"old": "old, in OpenSURF's order"}))
```

```{code-cell} ipython3
def shifted(rect, d):
    return tuple(v + d for v in rect)


rows = []
for lobe in (1, 3, 5, 9):
    R = rects(lobe)
    rows.append({
        "lobe": lobe,
        "new = dlib": R["new"] == R["dlib"],
        "new = OpenSURF, any order": sorted(R["new"]) == sorted(R["OpenSURF"]),
        "old = OpenSURF": R["old"] == R["OpenSURF"],
        "old = OpenSURF moved by (1, 1)":
            R["old"] == [shifted(x, 1) for x in R["OpenSURF"]],
    })
show_table(pd.DataFrame(rows))
```

The new code and dlib sum the same pixels for each call. Those are also
OpenSURF's boxes: SURF defines the filters, and the two libraries agree on
them. The old code sums OpenSURF's boxes moved one pixel down and to the
right. That is the shift of `on_hessian_det_approx.md` §4.2: its `_integ`
reads the integral image at `r` where OpenSURF reads `r - 1`.

The figure shows the `Dxy` boxes for lobe 3, and the point that each call
names: the centre for dlib and the new code, the top-left pixel for
OpenSURF.

```{code-cell} ipython3
def draw_calls(ax, rects, points, colour, title):
    """Outlines of `rects` and the named point of each call."""
    n = 11
    h = n // 2
    ax.set_xlim(-h - 0.5, h + 0.5)
    ax.set_ylim(h + 0.5, -h - 0.5)
    ax.set_xticks(range(-h, h + 1, 2))
    ax.set_yticks(range(-h, h + 1, 2))
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for top, left, bottom, right in rects:
        ax.add_patch(plt.Rectangle((left - 0.5, top - 0.5), right - left + 1,
                                   bottom - top + 1, fill=False, edgecolor=INK,
                                   linewidth=1))
    rr, cc = zip(*points)
    ax.plot(cc, rr, "o", color=colour, markersize=8, markeredgecolor="white")
    ax.plot(0, 0, "+", color=INK, markersize=10)
    ax.set_title(title)
    ax.set_xlabel("column offset $c$")
    ax.set_ylabel("row offset $r$")


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
dxy = slice(4, 8)
draw_calls(axes[0], RECTS["new"][dxy], [call[:2] for call in CALLS["new"][dxy]],
           COLOURS["centred"], "dlib and new: each call names the centre")
draw_calls(axes[1], RECTS["OpenSURF"][dxy], [call[:2] for call in CALLS["OpenSURF"][dxy]],
           COLOURS["top-left"], "OpenSURF: each call names the top-left pixel")
```

The boxes are the same in both panels. The points differ: the new code
names each box as dlib does.

(sec-port-names)=
## 6. Names

The names in each code, without comments, strings and docstrings. A leading
underscore is dropped, so that `_centered_rect` matches `centered_rect`.
The table lists every name of the old or the new code, and says where else
it occurs:

```{code-cell} ipython3
NAMES = {
    "OpenSURF": cpp_names(*BODIES["OpenSURF"].values()),
    "dlib": cpp_names(*BODIES["dlib"].values()),
    "old": python_names(TEXT["old"]),
    "new": python_names(NEW_SOURCE),
}


def origin(name):
    """Which of the two libraries use `name`."""
    found = [lib for lib in ("OpenSURF", "dlib") if name in NAMES[lib]]
    return " and ".join(found) or "neither"


rows = [{"name": name, "old": name in NAMES["old"], "new": name in NAMES["new"],
         "in": origin(name)}
        for name in sorted(NAMES["old"] | NAMES["new"], key=str.lower)]
names_table = pd.DataFrame(rows)
show_table(names_table)
```

```{code-cell} ipython3
summary = (names_table.melt(id_vars=["name", "in"], value_vars=["old", "new"],
                            var_name="code", value_name="uses")
           .query("uses")
           .groupby(["code", "in"])["name"]
           .apply(lambda s: ", ".join(sorted(s, key=str.lower)))
           .unstack(fill_value=""))
show_table(summary.reset_index())
```

The names show the same split as the calls, but they are weaker evidence:

* The new code uses dlib's names for the filter geometry (`lobe_size`,
  `area_inv`, `lobe_offset`), for the box functions (`centered_rect`), and
  for the corners of the box sum (`top_left`, `top_right`, `bottom_left`).
  It uses no name that only OpenSURF uses. `Dxx`, `Dyy` and `Dxy` are in
  both libraries, from the SURF paper.
* The old code uses OpenSURF's `w` for the filter size, as OpenSURF does,
  and the corner names `r2` and `c2` of `BoxIntegral`. Its `rl` matches by
  chance: in the old code it is a number of rows, in OpenSURF a response
  layer.
* The old code also uses `tl`, `tr`, `bl`, `br` and `x`, which are in dlib.
  In the old code the first four name box sums, and `x` is the argument of
  `_clip`. In dlib they name the offsets of the `Dxy` boxes and the
  argument of `centered_rect`.
* Short names such as `r`, `c` and `img` are in both libraries, and say
  nothing about the source.

## 7. The same values

The calls are dlib's, so the values must be too. `dlib_det` evaluates
dlib's three statements at each pixel, with a `sum_box` that adds the image
over the rectangle directly. It does not use an integral image, and, as
dlib requires, it refuses a box outside the image. So `dlib_det` is
defined only away from the border.

```{code-cell} ipython3
def dlib_components(image, lobe, r, c):
    """dlib's `Dxx`, `Dyy`, `Dxy` at pixel `(r, c)`, normalised."""
    def sum_box(rect):
        top, left, bottom, right = rect
        assert top >= 0 and left >= 0
        assert bottom < image.shape[0] and right < image.shape[1]
        return image[top:bottom + 1, left:right + 1].sum()

    body = BODIES["dlib"]["build_pyramid"]
    env = dlib_env(lobe, r, c, sum_box)
    area_inv = eval(as_python(statement(body, "area_inv"), integer=False),
                    {"lobe_size": lobe, "std": np})
    return [eval(as_python(statement(body, name)), env) * area_inv
            for name in ("Dxx", "Dyy", "Dxy")]


def dlib_det(image, lobe, margin):
    """dlib's determinant, with weight 0.81 and no clamp, inside `margin`."""
    out = np.full(image.shape, np.nan)
    for r in range(margin, image.shape[0] - margin):
        for c in range(margin, image.shape[1] - margin):
            Dxx, Dyy, Dxy = dlib_components(image, lobe, r, c)
            out[r, c] = Dxx * Dyy - 0.81 * Dxy * Dxy
    return out
```

The new code, run as Python from the copy, on signed noise of a shape that
is not square:

```{code-cell} ipython3
noise = np.random.default_rng(0).normal(size=(30, 34))
integral = noise.cumsum(0).cumsum(1)
rows = []
for lobe in (1, 3, 5, 7):
    margin = int(np.ceil(3 * lobe / 2))
    inner = (slice(margin, -margin),) * 2
    new = NEW["_hessian_matrix_det"](integral, lobe)
    reference = dlib_det(noise, lobe, margin)
    rows.append({"lobe": lobe, "pixels compared": str(reference[inner].size),
                 "max |new − dlib| / max |dlib|":
                     np.abs(new[inner] - reference[inner]).max()
                     / np.abs(reference[inner]).max()})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

mahotas runs a port of the same dlib code. It keeps dlib's clamp of
negative determinants to zero and dlib's sign of the Laplacian, and it
changes the weight from 0.81 to 0.36. The cell applies the same clamp, sign
and weight to dlib's statements, and compares the result with mahotas's
planes. In octave 0,
interval $i$ has lobe $2(i + 1) + 1$, and mahotas leaves zero a border
of $\lceil 3 (2 n + 3) / 2 \rceil$ pixels for $n$ intervals.

```{code-cell} ipython3
INTERVALS = 3
image = np.random.default_rng(1).uniform(size=(40, 44))
planes = mahotas_surf.pyramid(mahotas_integral(image.copy()), 1, INTERVALS, 1)[0]
margin = int(np.ceil(3 * (2 * INTERVALS + 3) / 2))
inner = (slice(margin, -margin),) * 2
rows = []
for i, plane in enumerate(planes):
    lobe = 2 * (i + 1) + 1
    expected = np.zeros(image.shape)
    for r in range(margin, image.shape[0] - margin):
        for c in range(margin, image.shape[1] - margin):
            Dxx, Dyy, Dxy = dlib_components(image, lobe, r, c)
            sign = -1.0 if Dxx + Dyy < 0 else 1.0
            expected[r, c] = sign * max(Dxx * Dyy - 0.36 * Dxy * Dxy, 0.0)
    rows.append({"lobe": lobe,
                 "max |mahotas − dlib statements| / max":
                     np.abs(plane[inner] - expected[inner]).max()
                     / np.abs(expected[inner]).max()})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

Both agree to rounding error. So the new code computes what dlib's
statements compute, and dlib's statements, run as Python here, compute what
an independent build of dlib's code computes.

## 8. What is not from dlib

The new code changes dlib in five ways. Its header lists them, and the
cell below measures the last four:

* it computes one lobe from `sigma` (`_surf_lobe`), not a pyramid of octaves
  and intervals;
* it computes every pixel, where dlib leaves a border;
* `_sum_of_area` clips each box to the image, so pixels outside the image
  count as zero, where dlib's `get_sum_of_area` requires the box to be
  inside the image;
* it does not clamp negative determinants to zero;
* it does not multiply by the sign of the Laplacian.

```{code-cell} ipython3
lobe = 5
margin = int(np.ceil(3 * lobe / 2))
new = NEW["_hessian_matrix_det"](integral, lobe)
padded = np.pad(noise, 3 * lobe)
from_padded = NEW["_hessian_matrix_det"](padded.cumsum(0).cumsum(1), lobe)[
    3 * lobe:-3 * lobe, 3 * lobe:-3 * lobe]
lobes = {s: NEW["_surf_lobe"](s) for s in (0.3, 1.9, 2, 3.9, 4, 7.2)}
show_table(pd.DataFrame([
    {"property": "pixels computed, of all pixels",
     "value": f"{np.isfinite(new).sum()} of {new.size}"},
    {"property": "border equals zero padding: max |difference|",
     "value": f"{np.abs(new - from_padded).max():.1e}"},
    {"property": "negative values kept",
     "value": f"{(new < 0).sum()} pixels below zero"},
    {"property": "lobe for sigma", "value": str(lobes)},
]))
```

`_surf_lobe` is in neither library: OpenSURF and dlib each fix a list of
lobes, and scikit-image chooses a lobe from `sigma`. The rule for it comes
from `on_hessian_det_approx.md` §9.1.

## 9. Summary

| question | old code (`main`) | new code (`surf-fixes`) | evidence |
| --- | --- | --- | --- |
| box calls | OpenSURF's arguments, at every lobe | dlib's arguments, in dlib's order, at every lobe | §4 |
| calls shared with the other library | none with dlib | none with OpenSURF | §4 |
| pixels summed | OpenSURF's boxes moved by (1, 1) | dlib's boxes, which are OpenSURF's boxes | §5 |
| names | OpenSURF's `w`, `r2`, `c2` | dlib's `lobe_size`, `area_inv`, `lobe_offset`, `centered_rect`, corner names; no OpenSURF-only name | §6 |
| values | | dlib's statements to rounding; mahotas, with its changes, to rounding | §7 |
| changes from dlib | | one lobe from `sigma`, every pixel, boxes clipped to the image (zero outside), no clamp, no sign | §8 |

The new code is a port of dlib: its box calls are dlib's, argument for
argument and in order, and it takes its names from dlib. It keeps none of
the forms that tie the old code to OpenSURF: the top-left-and-extent box
calls, the offsets such as `r - l + 1`, the names `b`, `l`, `w` and
`inverse_area`, and the `max(0, …)` clamp in the box sum. Its output is
still OpenSURF's at every pixel inside the image, because both libraries
compute the filters that SURF defines.

### Limits

* The notebook compares structure: calls, arguments, order and names. It
  cannot show how the code was written. The agent that wrote the port had
  read the OpenSURF source, and an earlier version of the fix
  (`on_hessian_det_approx.md` §9.1) kept OpenSURF's box sum, with its
  `min(row, height) - 1` form. The port replaced that version. This is a
  port of dlib, not a clean-room implementation.
* Whether the old code, in releases up to now, needs any action is a
  question for the maintainers, not a result of this notebook.
* The new code is a copy of the uncommitted branch file, identified by its
  hash. The comparators are fixed at the commits in §1.
* The C++ statements run in Python. That is faithful only for the
  operations they use: integer `/` on positive values, `+`, `-`, `*`, and
  the calls. dlib's `get_sum_of_area` and OpenSURF's `BoxIntegral` do not
  run; their rectangles come from their stated rules, and §7 checks dlib's
  against mahotas.
* §6 compares names exactly, with case. The old code's `dxx` does not match
  `Dxx`.
