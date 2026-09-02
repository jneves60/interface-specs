# Folding

Folding is the mechanism by which a single parameterised SDSC artifact describes
the behaviour of all active cores without duplicating data for every core, corelet,
or time step. Instead of storing separate values for each combination of indices,
a `FoldManager` encodes a compact expression — constant, lookup table, or affine
formula — that is evaluated at compile time or corrected just-in-time before the
job is launched.

The two primitive building blocks are:

- [`FoldProperty`](foldproperty.md) — pairs a **fold factor** (number of slices) with a
  **label** (hierarchy level name such as `"core"`, `"corelet"`, `"time"`).
- [`FoldManager`](foldmanager.md) — a parallel pair of arrays (`dim_prop_func`,
  `dim_prop_attr`) plus an optional lookup table (`data_`) that together encode
  a multi-dimensional mapping from index tuples to concrete values.

For the full field-level reference of each type, follow the links above.

## Where Folding Appears

`FoldManager` objects are used in three distinct roles across the schema:

| Field | Role |
|---|---|
| `SuperDsc.sdscFolds_` | Bundle-level fold addresses and mappings |
| `ScheduleTreeNode.startAddressCoreCorelet_` | Per-core / per-corelet tensor start addresses |
| `CoordinateInfo.folds` | Per-dimension coordinate stride computation |

## Function Types

Each entry in `dim_prop_func` is a single-key object that controls how the loop
index for that dimension resolves to a value:

| Key | Structure | Behaviour |
|---|---|---|
| `Const` | `{"Const": {}}` | Returns the same value for all indices. The single `data_` entry (keyed `"[0]"`) is used regardless of the loop counter. |
| `Map` | `{"Map": {}}` | Looks up the value in `data_` using the fold-coordinate tuple as the key. Used for arbitrary per-core mappings, e.g. start addresses. |
| `Affine` | `{"Affine": {"alpha_": <int>, "beta_": <int>}}` | Computes `alpha_ * index + beta_`. No `data_` lookup required. Used for coordinates that follow a linear stride. |
| `WkSplit` | `{"WkSplit": {}}` | Returns the work-slice index assigned to this core for this dimension, resolved from `coreIdToWkSlice_`. |

`dim_prop_func` and `dim_prop_attr` must always have the same length — one entry per fold dimension.

## scheduleTree.startAddressCoreCorelet_

This field stores the starting memory address for each core and corelet for a
given tensor allocation. The typical `dim_prop_func` pattern uses `Map` for the
core dimension (each core may have a different start address) and `Const` for the
corelet and time dimensions (all corelets on a core share the same base address):

```json
"startAddressCoreCorelet_": {
  "dim_prop_func": [
    {"Map":   {}},
    {"Const": {}},
    {"Const": {}}
  ],
  "dim_prop_attr": [
    {"factor_": 20, "label_": "core"},
    {"factor_": 1,  "label_": "corelet"},
    {"factor_": 1,  "label_": "time"}
  ],
  "data_": {
    "[0, 0, 0]":  "0",   "[1, 0, 0]":  "0",   "[2, 0, 0]":  "0",
    "[3, 0, 0]":  "0",   "[4, 0, 0]":  "0",   "[5, 0, 0]":  "128",
    "[6, 0, 0]":  "128", "[7, 0, 0]":  "128", "[8, 0, 0]":  "128",
    "[9, 0, 0]":  "128", "[10, 0, 0]": "256", "[11, 0, 0]": "256",
    "[12, 0, 0]": "256", "[13, 0, 0]": "256", "[14, 0, 0]": "256",
    "[15, 0, 0]": "384", "[16, 0, 0]": "384", "[17, 0, 0]": "384",
    "[18, 0, 0]": "384", "[19, 0, 0]": "384"
  }
}
```

The example above corresponds to a tensor with a 4-way split. Each of the 4
slices is consumed by 5 cores, so the start address is the same for 5
consecutive cores. Each slice is 128 bytes, so successive slice start addresses
are 128 bytes apart.

Start addresses in `data_` can be **symbolic**: instead of a concrete byte
offset, a symbolic identifier is used as the value. The actual address is
substituted just before the job is launched. Symbolic bindings are declared in
the bundle MLIR file — see [MLIR Bundle API](MLIR-bundle-API.md).

## scheduleTree.coordinates_.coordInfo.\<dim name\>.folds

Under `scheduleTree_`, the `folds` field inside each `CoordinateInfo` entry
describes how a tensor dimension is progressively split across cores, corelets,
rows, and the final leaf entities. The `dim_prop_attr` labels follow a fixed
hierarchy:

| Label | Level |
|---|---|
| `core_fold` | Split across cores |
| `corelet_fold` | Split across corelets within a core |
| `row_fold` | Split across rows within a corelet |
| `elem_arr_0` | Element count in the innermost (leaf) slice |
| `elem_arr_1` | Number of sticks per slice (stick dimensions only) |

The `padding` field of each `CoordinateInfo` entry takes one of two values:
`"nopad"` (no padding applied to this dimension) or `"pad"` (padding is applied).

`coordInfo.spatial` is typically `3`, indicating three spatial split levels
(core, corelet, row). `coordInfo.temporal` is set to `0` by the frontend.
`coordInfo.elemArr` is `1` for non-stick dimensions and `2` for stick dimensions.

The general structure is:

```json
"coordinates_": {
  "coordInfo": {
    "<dim_name>": {
      "spatial":  3,
      "temporal": 0,
      "elemArr":  1,
      "padding":  "nopad",
      "folds": {
        "dim_prop_func": [
          {"Affine": {"alpha_": <cores spanned per core-wise split>, "beta_": 0}},
          {"Affine": {"alpha_": 1, "beta_": 0}},
          {"Affine": {"alpha_": 1, "beta_": 0}},
          {"Affine": {"alpha_": <size of leaf slice>, "beta_": 0}}
        ],
        "dim_prop_attr": [
          {"factor_": <number of core-wise splits>,    "label_": "core_fold"},
          {"factor_": <number of corelet splits>,      "label_": "corelet_fold"},
          {"factor_": <number of row splits>,          "label_": "row_fold"},
          {"factor_": <number of elements per slice>,  "label_": "elem_arr_0"}
        ]
      }
    }
  }
}
```

For a **stick dimension** `coordInfo.elemArr` is `2` and `dim_prop_attr`
includes an additional `elem_arr_1` entry. The `factor_` of `elem_arr_0` gives
the number of elements per stick; the `factor_` of `elem_arr_1` gives the
number of sticks in the slice.

### Worked example

Consider a tensor with dimensions `x`, `in`, and `out` of sizes `[4, 2880, 2880]`.
The `x` dimension has a 4-way split across cores; `out` has a 5-way split
across cores; `in` is not split. Total splits: 20, one per core.

#### Dimension **x** (4-way core split, non-stick)

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 4, "label_": "core_fold"},
    {"factor_": 1, "label_": "corelet_fold"},
    {"factor_": 1, "label_": "row_fold"},
    {"factor_": 1, "label_": "elem_arr_0"}
  ]
}
```

#### Dimension **out** (5-way core split, stick dimension)

`coordInfo.elemArr` is `2`. `dim_prop_attr` includes `elem_arr_1` for the
number of sticks per slice.

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 5,   "label_": "core_fold"},
    {"factor_": 1,   "label_": "corelet_fold"},
    {"factor_": 1,   "label_": "row_fold"},
    {"factor_": 64,  "label_": "elem_arr_0"},
    {"factor_": 9,   "label_": "elem_arr_1"}
  ]
}
```

`elem_arr_0 factor_ = 64` → 64 elements per stick.
`elem_arr_1 factor_ = 9` → 9 sticks per core slice (9 × 64 = 576 = 2880 / 5).

#### Dimension **in** (not split)

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 2880, "beta_": 0}},
    {"Affine": {"alpha_": 0,    "beta_": 0}},
    {"Affine": {"alpha_": 0,    "beta_": 0}},
    {"Affine": {"alpha_": 1,    "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 1,    "label_": "core_fold"},
    {"factor_": 1,    "label_": "corelet_fold"},
    {"factor_": 1,    "label_": "row_fold"},
    {"factor_": 2880, "label_": "elem_arr_0"}
  ]
}
```

`core_fold factor_ = 1` confirms no core-level splitting. `alpha_ = 2880` at
the core level means the full 2880-element span is the coordinate stride passed
to the fold engine.

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: FoldProperty →](foldproperty.md) |
|:--|:--:|--:|
