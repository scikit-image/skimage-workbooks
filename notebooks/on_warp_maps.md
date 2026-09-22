---
title: On the maps you can hand to `warp`
date: 2026-09-14
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

`skimage.transform.warp` takes a second argument, `inverse_map`, that accepts
four different kinds of object: a geometric transform, a bound `.inverse`
method, a bare `(3, 3)` array, a callable, and an array of coordinates. This
notebook shows what each one means, where they agree, and where they quietly
do not.

It is not about the name `inverse_map`. It is about the four things the
parameter accepts.

Coordinates are in **array order** unless a cell says otherwise: the first
number indexes the first array axis, which runs down the picture. Where `warp`
uses the other order, that is one of the findings.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
```

```{code-cell} ipython3
from skimage.transform import warp, warp_coords, SimilarityTransform, ProjectiveTransform
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs:
# worst CVD dE 9.2, worst normal-vision dE 24.0.
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
C_THREE = "#1baf7a"
C_OFF = "#f2f1ec"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)
```

+++

## Drawing helpers

```{code-cell} ipython3
def pixel_axes(ax, shape, title=None):
    """An empty pixel grid, axis 0 down and axis 1 right."""
    n_i, n_j = shape
    ax.set_xlim(-0.5, n_j - 0.5)
    ax.set_ylim(n_i - 0.5, -0.5)
    ax.set_xticks(range(0, n_j, 2))
    ax.set_yticks(range(0, n_i, 2))
    ax.set_xticks(np.arange(n_j + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_i + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.7)
    ax.tick_params(which="both", length=0, labelsize=7, colors=MUTED)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def fill(ax, image, color, threshold=0.5):
    """Fill every pixel of `image` above `threshold`."""
    for i, j in np.argwhere(image > threshold):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                               edgecolor="white", linewidth=0.8, zorder=1))
    return ax


def spot(shape=(9, 9), at=(2, 3)):
    """A single bright pixel, so a displacement is unmistakable."""
    img = np.zeros(shape)
    img[at] = 1.0
    return img
```

+++

## 1. The problem: one displacement, five spellings

Take a single bright pixel at `(2, 3)` and move the content **down two rows**,
to `(4, 3)`. Every kind of argument can express that. They do not look alike.

```{code-cell} ipython3
img = spot()
ii, jj = np.indices(img.shape, dtype=float)

spellings = {
    "transform .inverse": SimilarityTransform(translation=(0, 2)).inverse,
    "transform object": SimilarityTransform(translation=(0, -2)),
    "(3, 3) matrix": np.array([[1.0, 0, 0], [0, 1, -2], [0, 0, 1]]),
    "callable": lambda c: c - np.array([0, 2]),
    "coordinate array": np.array([ii - 2, jj]),
}

fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.3))
for ax, (name, arg) in zip(axes, spellings.items()):
    out = warp(img, arg)
    pixel_axes(ax, img.shape, name)
    fill(ax, img, C_OFF)
    fill(ax, out, C_ONE)
fig.suptitle("five arguments, one displacement (pale = before, blue = after)", y=1.04)
fig.tight_layout()
```

```{code-cell} ipython3
for name, arg in spellings.items():
    out = warp(img, arg)
    print(f"{name:<20} bright pixel now at {tuple(int(x) for x in np.argwhere(out > 0.5)[0])}")
```

All five agree. The rest of this notebook is about what you had to know to
write each one.

+++

## 2. The contract they share

Every kind answers the same question: **for this output pixel, where in the
input image do I sample?** So the map runs from output coordinates to input
coordinates.

That is easy to assert and easy to get backwards, so measure it. Give the
output a different shape from the input, then record what a callable is
handed.

```{code-cell} ipython3
seen = {}

def spy(coords):
    """Record what warp passes in, then act as the identity."""
    seen["shape"] = coords.shape
    seen["min"], seen["max"] = coords.min(axis=0), coords.max(axis=0)
    return coords


warp(np.zeros((9, 9)), spy, output_shape=(5, 20))
print("input image shape (9, 9), output_shape (5, 20)")
print(f"   the callable was handed {seen['shape']}")
print(f"   component 0 spans {seen['min'][0]:.0f} .. {seen['max'][0]:.0f}")
print(f"   component 1 spans {seen['min'][1]:.0f} .. {seen['max'][1]:.0f}")
print()
print("an input grid would span 0..8 in both; an output grid spans 0..19 and 0..4")
```

100 is 5 times 20. The callable is walked over the **output** grid and must
return **input** coordinates. The direction is fixed for every kind of
argument.

+++

## 3. The coordinate array

The array is that map written out in full: its **shape** spans the output, and
its **values** are coordinates in the input image. Point every entry at one
input pixel and the whole output takes that pixel's value.

```{code-cell} ipython3
ramp = np.arange(81, dtype=float).reshape(9, 9)
coords = np.zeros((2, 5, 20))
coords[0], coords[1] = 2, 3          # every output pixel samples input (2, 3)
out = warp(ramp, coords)
print(f"ramp[2, 3] = {ramp[2, 3]}")
print(f"array shape {coords.shape} -> output shape {out.shape}")
print(f"every output value is {np.unique(out)}")
```

This is the only kind of argument that carries no convention of its own. It is
plain array-order indices, and it is unaffected by any change to the
coordinate convention elsewhere in the library.

+++

## 4. The callable, and the axis order that flips

The callable is the same map, computed rather than tabulated. `warp_coords`
converts one into the other, and the two routes agree exactly.

```{code-cell} ipython3
def f(c):
    return c - np.array([0, 2])


print("warp(img, f) == warp(img, warp_coords(f, shape)):",
      np.allclose(warp(img, f), warp(img, warp_coords(f, img.shape))))
```

But the two forms do not use the same axis order. The callable receives
`(column, row)` pairs. The coordinate array is indexed `(row, column)`. So the
same expression, written the same way, means different things.

```{code-cell} ipython3
same_expression = {
    "callable: c - [0, 2]": lambda c: c - np.array([0, 2]),
    "array: [ii, jj - 2]": np.array([ii, jj - 2]),
}
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
for ax, (name, arg) in zip(axes, same_expression.items()):
    out = warp(img, arg)
    at = tuple(int(x) for x in np.argwhere(out > 0.5)[0])
    pixel_axes(ax, img.shape, f"{name}\n-> {at}")
    fill(ax, img, C_OFF)
    fill(ax, out, C_TWO)
fig.suptitle("subtract 2 from component 1, both ways", y=1.06)
fig.tight_layout()
```

One moves the content down, the other moves it right. `warp_coords` is the
bridge that performs the flip, which is why the two routes still agree when you
go through it.

+++

## 5. The transform object

A transform object is the only kind that carries a direction of its own, and
`warp` uses it as the output-to-input map. So a transform built to describe
where the content should *go* has to be inverted first.

```{code-cell} ipython3
down_two = SimilarityTransform(translation=(0, 2))   # (x, y): move content down
for name, arg in (("passed directly", down_two), ("passed as .inverse", down_two.inverse)):
    at = tuple(int(x) for x in np.argwhere(warp(img, arg) > 0.5)[0])
    print(f"SimilarityTransform(translation=(0, 2)) {name:<19} -> {at}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
for ax, (name, arg) in zip(axes, (("passed directly", down_two),
                                  ("passed as .inverse", down_two.inverse))):
    out = warp(img, arg)
    at = tuple(int(x) for x in np.argwhere(out > 0.5)[0])
    pixel_axes(ax, img.shape, f"{name}\n-> {at}")
    fill(ax, img, C_OFF)
    fill(ax, out, C_THREE)
fig.suptitle("SimilarityTransform(translation=(0, 2)), two ways", y=1.06)
fig.tight_layout()
```

The object and its inverse are both accepted, both are spelled almost the same
way at the call site, and they move the content in opposite directions. Which
one is correct depends entirely on what the caller meant when they built it.

+++

## 6. The bare matrix

+++

### What a homogeneous matrix is

A `D`-dimensional affine map has two parts: a linear part that can rotate,
scale and shear, and a translation that can only be added. Matrix
multiplication alone cannot add a constant, so the coordinate is **augmented**
with a trailing `1`, and the matrix is grown to `(D + 1, D + 1)` to match:

```
    | a  b  tx |   | x |     | a*x + b*y + tx |
    | c  d  ty | . | y |  =  | c*x + d*y + ty |
    | p  q   s |   | 1 |     | p*x + q*y + s  |
```

The blocks have distinct jobs:

- the top-left `D` by `D` block is the linear part;
- the last **column** is the translation, which the augmented `1` turns into an
  addition;
- the last **row** is the projective part. For an affine map it is
  `(0, ..., 0, 1)`, so the trailing component stays `1` and can be discarded.
  When it is anything else, the result is divided by that component, which is
  what makes perspective possible.

The two roles are visible directly. An entry in the last column shifts one
axis:

```{code-cell} ipython3
for entry, label in (((0, 2), "M[0, 2]"), ((1, 2), "M[1, 2]")):
    m = np.eye(3)
    m[entry] = -2
    out = warp(img, m)
    print(f"{label} = -2 moves the pixel from (2, 3) to "
          f"{tuple(int(x) for x in np.argwhere(out > 0.5)[0])}")
```

So the first homogeneous row is the **column** axis and the second is the
**row** axis: the matrix is written in `(x, y)` order, like the callable of
section 4 and unlike the coordinate array of section 3.

An entry in the last row divides, and evenly spaced inputs stop being evenly
spaced:

```{code-cell} ipython3
projective = np.eye(3)
projective[2, 0] = 0.05           # the trailing component now depends on x
points = np.array([[0.0, 0], [10, 0], [20, 0]])       # (x, y) row vectors

print("input  x:", points[:, 0])
print("output x:", np.round(ProjectiveTransform(projective)(points)[:, 0], 3))
print("an affine matrix would have left the spacing alone")
```

+++

### It is recognised by its shape, and the shape is ambiguous

`warp` decides that an array is a matrix by testing `shape == (3, 3)`. It never
consults the image, so a 3-D image gets the same answer as a 2-D one.

```{code-cell} ipython3
cube = np.arange(125, dtype=float).reshape(5, 5, 5)
print("2-D image + (3, 3) array -> homography:",
      tuple(int(x) for x in np.argwhere(warp(img, np.eye(3)) > 0.5)[0]))
print("3-D image + (3, 3) array -> output shape", warp(cube, np.eye(3)).shape)
```

That second line is the problem. For a 3-D image a `(3, 3)` array is also a
perfectly good coordinate array: three components over an output of shape
`(3,)`, which would have produced an output of shape `(3,)`. There is no way to
ask for it.

```{code-cell} ipython3
coords_3 = np.zeros((3, 3))
coords_3[:, 0] = [1, 2, 3]        # three components, output shape (3,)
print("meant as coordinates, output shape would be (3,); got",
      warp(cube, coords_3).shape)
```

The collision is not between the two kinds of array. It is between two readings
of the same image: `warp` treats any 3-dimensional input as 2-D with channels,
because it has no way to be told otherwise.

```{code-cell} ipython3
rgb = np.zeros((9, 9, 3))
rgb[2, 3, :] = 1.0
shift = np.eye(3)
shift[1, 2] = -2
print("(9, 9, 3) as 2-D with channels ->",
      tuple(int(x) for x in np.argwhere(warp(rgb, shift)[..., 0] > 0.5)[0]))
print("(5, 5, 5) is read the same way, though it is a volume")
```

+++

### With `channel_axis`, the ambiguity disappears

Suppose `warp` gains a `channel_axis`, as the rest of the library has. Then the
**spatial** dimensionality `S` is known rather than guessed, and the two kinds
of array separate by their leading axis alone:

```{code-cell} ipython3
print(f"{'S':>3}{'homogeneous matrix':>22}{'coordinate array':>24}{'leading axes':>14}")
for S in range(1, 6):
    print(f"{S:>3}{str((S + 1, S + 1)):>22}{f'({S}, *output_shape)':>24}"
          f"{f'{S + 1} vs {S}':>14}")
```

A matrix leads with `S + 1` and coordinates lead with `S`. They differ by one
at every dimensionality, so the test is exact and needs no special case:

```{code-cell} ipython3
def kind_of(arr, image, channel_axis=None):
    """Is this array a homogeneous matrix, or an array of coordinates?"""
    spatial = image.ndim - (channel_axis is not None)
    if arr.shape == (spatial + 1, spatial + 1):
        return "homogeneous matrix"
    if arr.shape[0] == spatial:
        return "coordinate array"
    raise ValueError(
        f"array of shape {arr.shape} is neither a "
        f"{(spatial + 1, spatial + 1)} matrix nor coordinates for {spatial} axes"
    )


cases = [
    ("(9, 9) image,      (3, 3) array", np.eye(3), img, None),
    ("(9, 9, 3) rgb,     (3, 3) array", np.eye(3), rgb, -1),
    ("(5, 5, 5) volume,  (3, 3) array", np.eye(3), cube, None),
    ("(5, 5, 5) volume,  (4, 4) array", np.eye(4), cube, None),
]
for label, arr, image, axis in cases:
    print(f"{label:<34} -> {kind_of(arr, image, axis)}")
```

The same `(3, 3)` array is a matrix for an RGB image and a coordinate array for
a volume. `channel_axis` is the whole of what distinguishes them.

+++

### And then the matrix stops being 2-D only

Once `S` is known, nothing about a homogeneous matrix is specific to two
dimensions: `(4, 4)` describes a 3-D affine map in the same way `(3, 3)`
describes a 2-D one. What blocks it today is not the matrix but `warp_coords`,
which is written for two axes. Given a coordinate builder that is not, a
`(4, 4)` matrix behaves as it should.

```{code-cell} ipython3
import scipy.ndimage as ndi
from skimage.transform import AffineTransform


def warp_coords_nd(coord_map, shape):
    """`warp_coords` with no dimensionality assumption, in array order."""
    coords = np.indices(shape, dtype=float).reshape(len(shape), -1).T
    return coord_map(coords).T.reshape((len(shape),) + tuple(shape))


volume = np.zeros((8, 9, 10))
volume[2, 3, 4] = 1.0
M4 = np.eye(4)
M4[:3, 3] = [1, 0, 0]              # shift one step along the first axis

moved = ndi.map_coordinates(volume, warp_coords_nd(AffineTransform(matrix=M4),
                                                   volume.shape), order=1)
print("(4, 4) matrix moves the voxel to",
      [tuple(int(x) for x in p) for p in np.argwhere(moved > 0.5)], "from (2, 3, 4)")
print("scipy.ndimage.affine_transform agrees:",
      [tuple(int(x) for x in p)
       for p in np.argwhere(ndi.affine_transform(volume, M4, order=1) > 0.5)])
```

Section 8 shows what `warp` does with that same `(4, 4)` matrix today.

## 7. What each kind supports

The kinds are not interchangeable in what they accept alongside them.

```{code-cell} ipython3
def shift_rows(c, dy=0):
    return c - np.array([0, dy])


def moved_to(arg, **kwargs):
    """Where the bright pixel ends up."""
    out = warp(img, arg, **kwargs)
    return tuple(int(x) for x in np.argwhere(out > 0.5)[0])


print("callable + map_args         ->", moved_to(shift_rows, map_args={"dy": 2}))
print("coordinate array + map_args ->",
      moved_to(np.array([ii, jj]), map_args={"dy": 2}), " (ignored)")
```

`map_args` is meaningful only for the callable. Passing it with any other kind
is silently accepted and does nothing, except that it also disables the fast
Cython path.

+++

## 8. Three dimensions

Only the coordinate array works in N-D. The others are documented as 2-D, but
the failure is not always a message.

```{code-cell} ipython3
from skimage.transform import AffineTransform

M4 = np.eye(4)
M4[:3, 3] = [1, 0, 0]
t3 = AffineTransform(matrix=M4)
vol = np.zeros((8, 9, 10))
vol[2, 3, 4] = 1.0

grid = np.indices(vol.shape, dtype=float)
grid[0] -= 1
print("coordinate array, 3-D  ->",
      [tuple(int(x) for x in p) for p in np.argwhere(warp(vol, grid) > 0.5)])

out = warp(vol, t3, order=1)
found = np.argwhere(out > 0.5)
print("3-D transform, order=1 ->",
      [tuple(int(x) for x in p) for p in found], " (nothing: the voxel is gone)")
try:
    warp(vol, t3, order=0)
except Exception as e:
    print("3-D transform, order=0 ->", type(e).__name__)
try:
    warp(vol, M4)
except Exception as e:
    print("(4, 4) matrix          ->", type(e).__name__, "-", str(e)[:44])
```

A 3-D transform at `order=1` returns an empty array with no warning: `warp`
reads a 3-D image as 2-D-plus-channels and hands the `(4, 4)` matrix to a
routine that reads the first nine of its sixteen values.

+++

## 9. Refactoring issues

Setting the parameter's name aside, four things in the list above are
properties of the design rather than of any one call.

### The kinds disagree about axis order

The callable takes `(column, row)`; the coordinate array is `(row, column)`.
Two spellings of one map, in opposite conventions, in one parameter. Section 4
shows the same expression producing a vertical move in one and a horizontal
move in the other. Only the array form is convention-free and therefore
untouched by a move to array order elsewhere.

### The matrix is recognised by shape, and the shape is ambiguous

Section 6 works this through. The ambiguity is not between the two kinds of
array — their leading axes differ by one at every dimensionality — but between
two readings of the image, because `warp` has no `channel_axis` and so treats
every 3-dimensional input as 2-D with channels.

Adding `channel_axis` settles it, and settles more than it looks. It removes
the shape test's special case, it makes the error message for a wrong shape
exact rather than generic, and it makes an `(S + 1, S + 1)` matrix meaningful
at any `S` — which is most of what a 3-D `warp` needs, the rest being a
`warp_coords` that carries no dimensionality assumption.

### Direction lives in the object, not in the call

Section 5: a transform and its `.inverse` are both accepted, look alike at the
call site, and move the content opposite ways. No other kind has this
property, because no other kind knows which way round it is.

### The kinds are not equally capable

`map_args` applies to one kind. N-D works for one kind. The fast path applies
to two. A reader cannot tell any of this from the signature, and section 8
shows one combination failing silently rather than raising.

### What this suggests

Dispatching on type rather than on shape removes the sniffing: a transform
object and a coordinate array are told apart by what they are, not by their
dimensions. That is Option A in section 11.4 of `coordinate_port_plan.md`, in
the separate [port-notes`
repository](https://github.com/matthew-brett/port-notes), which also covers
what to do with the bare matrix and the callable. Field usage of the four
kinds is measured in section 10.2 of that document; this notebook does not
re-derive it.

The axis-order split is the one item that the coordinate port fixes on its own:
once callables receive array-order coordinates, the callable and the array
agree, and `warp_coords` stops needing to flip anything.

+++

## Summary

| Kind | Direction | Axis order | `map_args` | N-D | Recognised by |
| --- | --- | --- | --- | --- | --- |
| transform object | output to input, but the object may be either way round | its own | no | no | type |
| `.inverse` method | output to input | its own | no | no | type and name |
| `(3, 3)` array | output to input | `(x, y)` | no | no | **shape alone** |
| callable | output to input | `(column, row)` | **yes** | no | callable |
| coordinate array | output to input | `(row, column)` | no | **yes** | ndarray |

Every kind answers the same question. They differ in axis order, in what may
accompany them, in dimensionality, and in how `warp` recognises them.

The table describes `warp` as it stands. With a `channel_axis`, two rows would
change: the `(3, 3)` array becomes an `(S + 1, S + 1)` array recognised by
shape *against a known spatial dimensionality* rather than by shape alone, and
its N-D column becomes yes.

Measured against the branch in this working tree, for 2-D inputs except where
section 8 says otherwise. Anti-aliasing, `order` above 1, `channel_axis`
handling and non-default `mode` are not examined here.
