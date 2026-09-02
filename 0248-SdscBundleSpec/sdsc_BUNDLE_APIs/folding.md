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

For full details on how these fields are used in `ScheduleTreeNode`, see
[`startAddressCoreCorelet_`](scheduletreenode.md#startaddresscorecorelet_) and
[`coordinates_`](scheduletreenode.md#coordinates_) in
[ScheduleTreeNode](scheduletreenode.md).

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: FoldProperty →](foldproperty.md) |
|:--|:--:|--:|
