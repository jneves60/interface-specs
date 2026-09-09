# Padding

For window/padded operations, such as convolution, padding information should be added to both `N_` and `dataStageParam_` in `sdsc.dscs_[0]`,
capturing information about front/back padding, stride, and related kernel dimension. If a padded dimension is chunked across cores,
front/back padding should be set to `-1` in core datastage.

When a dimension is padded due to window/padded operations like convolution, details of padding need to be specified via the following fields.

## (i) paddingSizes_

`paddingSizes_` is a sub-field of both `dcs[0].N_` and `dcs[0].dataStageParam_`. One entry per padded dimension; the key is the primary dimension name (e.g. `"r_"`, `"c_"`).

| Field | Type | Description |
|---|---|---|
| `padFront_` | integer | Front padding elements. Default `0`. Set to `-1` when the dimension is chunked across cores (see note above). |
| `padBack_` | integer | Back padding elements. Default `0`. Set to `-1` when the dimension is chunked across cores. |
| `unneededPad_` | integer | Total unneeded padding elements (sum of front and back unneeded). |
| `unneededPadFront_` | integer | Unneeded padding elements originating from `padFront_`. |
| `unneededPadBack_` | integer | Unneeded padding elements originating from `padBack_`. |
| `totalSize_` | integer or `"N/A"` | Total input size inclusive of padding. `"N/A"` when `windowDim_` is unset and `padFront_` < 0. Regardless of whether padding is applied, this corresponds to the input size after padding — e.g. for a 128×128 image with padding=1 this is `130`; without padding it is `128`. |
| `stride_` | integer ≥ 1 | Stride applied with the operation. Default `1`. |
| `dilation_` | integer ≥ 1 | Dilation applied with the operation. Default `1`. |
| `windowDim_` | string | Associated window primary dimension name (e.g. `"ki_"`). Uses `primaryDimToString` mapping. |

The dimension's size as specified in `N_` and `dataStageParam_` corresponds to the **output** size — e.g. `128` with padding=1 and `126` without padding for a 128×128 input.

## (ii) padding field in CoordinateInfo

The `padding` field inside `coordinates_.coordInfo[<dim>]` (a `CoordinateInfo` object within a `scheduleTree_` node) can take on the following values:

```
nopad
lowered_padded
padded_nozeropad
padded_wzeropad
padded_fullspan
padded_fullspan_wunneeded
```

**`padded_nozeropad`** — to be used with conv2d when padding is non-zero.
**`padded_fullspan_wunneeded`** — to be used with conv2d when padding is zero.

---

| [← Previous: FoldManager](foldmanager.md) | [↑ Table of Contents](README.md) | [Next: Stick Layout Constraints →](stick-layout-constraints.md) |
|:--|:--:|--:|
