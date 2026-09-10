# SDSC Bundle API — Change Log

This file records changes to the API documentation after each task or Bob session.
It is the working log that accumulates until a PR is merged.

**On PR merge:** review all entries since the last version tag, determine the
SemVer bump (MAJOR / MINOR / PATCH), update `Spec Version` and `Git Baseline`
in [README.md](README.md), and add a one-line summary entry to the
[Released Versions](#released-versions) table below.

> **How to add an entry during a task:**
> Prepend a new `### Task` block at the top of [Unreleased Changes](#unreleased-changes).
> Include the date, a brief description of what the task was, the files changed,
> and the commits involved.

---

## Unreleased Changes

### Task — Fix confusing scf.for loop-carried variable constraint wording (2026-09-09)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Merge two misleading bullets into one: "Loop-carried variables are not supported" and "Only the induction variable may be used directly inside the loop body" were confusing because the second implied other ops were banned. Replaced with a single bullet that states the constraint and clarifies the induction variable is freely usable in the loop body. |

---

## Released Versions

| Version | Date | Git Baseline | Summary |
|---|---|---|---|
| 1.0.0 | 2026-09-09 | `fcd337a` | Initial versioned baseline. All API docs brought to standard. See [v1.0.0 detail](#v100--2026-09-09) below. |

---

## v1.0.0 — 2026-09-09

Baseline version established from the full git history of the directory.
All commits prior to this version are recorded below, grouped by the task
that produced them.

---

### Task — Compliance review and schema/doc fixes (2026-09-09)

**Commits:** `fcd337a`, `06ae8f1`, `eb383ec`, `e816731`, `1a74153`, `ebeeb23`, `59ca249`

| File | Change |
|---|---|
| `padding.md` | Fix padding enum casing/field; consolidate schema files |
| `padding.md` | Restore original general intro paragraph |
| `padding.md` | Expand `paddingSizes_` with op-specific variants (avgpool2d, depthwise conv2d) and emission rules |
| `padding.md` | Document missing `unneededPad*` fields |
| `labeleddatastructure.md` | Remove stale `SDSC-json-api.md` xref from `wordLength` naming note |
| `scheduletreenode.md` | Fix `padding_` field description |
| `sdscbundle-schema.json` | Schema sync with doc fixes |

---

### Task — Link and reference fixes (2026-09-09 / 2026-09-08)

**Commits:** `a195245`, `f8c71ff`, `c16da13`

| File | Change |
|---|---|
| `reference.md` | Fix broken links |
| `Overview.md` | Fix SuperDSC-Bundle spec link |
| `Overview.md` | Remove redundant notes |

---

### Task — MLIR Bundle API corrections (2026-09-07)

**Commits:** `27e8788`, `b0f537b`, `b30de51`, `7860a50`, `0630ff8`, `66f603e`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Soften `device_mem_allocate` loop placement from MUST to SHOULD |
| `MLIR-bundle-API.md` | Correct `symbol_ids` scoping description |
| `MLIR-bundle-API.md` | Remove misplaced `scf.for` loop bound note from Symbolic Values section |
| `MLIR-bundle-API.md` | Remove implementation-detail parentheticals from `scf.for` descriptions |
| `MLIR-bundle-API.md` | Correct symbolic loop bounds description |
| `MLIR-bundle-API.md` | Broaden bundle container operands description to include symbolic dimension sizes |

---

### Task — Symbolic values documentation and MLIR example fixes (2026-09-05)

**Commits:** `33b5bad`, `57e023e`, `1e4d678`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Add symbolic values and addresses documentation; fix `device_mem_allocate` loop constraint contradiction |
| `MLIR-bundle-usage-examples.md` | Add symbolic address and dimension examples; fix example inconsistencies |
| `MLIR-complete-example.md` | Fix MLIR example inconsistencies |

---

### Task — MLIR Bundle API expansion (2026-09-04 / 2026-08-27)

**Commits:** `102848d`, `2f57043`, `512e04b`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Document bundle parameters; add `input_arg_extract`; expand operations table; reformat to formal API reference style |
| `MLIR-bundle-usage-examples.md` | Add `sdscbundle.device_mem_allocate` examples |

---

### Task — Reference and stage restructuring (2026-09-03 / 2026-09-02)

**Commits:** `36228ac`, `61cba1d`, `36f8bb4`, `e6a8ca8`, `34a52bc`, `ea55e63`, `de23860`, `a1ecdbc`

| File | Change |
|---|---|
| `reference.md` | Add KTIR RFC link; fix links to public repos; add torch-spyre; remove glossary section |
| `README.md` | Remove error handling and best practices stages; restructure into Introduction / MLIR / JSON stages; reorganize into staged learning order |
| `SDSC-json-api.md` | Add JSON object hierarchy diagram |
| Multiple | Sync markdown files with `sdscbundle-schema2.json` |

---

### Task — JSON object hierarchy and navigation (2026-09-02)

**Commits:** `8cacc1d`, `26d7566`, `e73d512`, `a1a114f`, `3d56c9e`

| File | Change |
|---|---|
| `JSON-object-Hierarchy.md` | Update to match `sdscbundle-schema2.json` |
| All | Add Previous navigation links to all file footers |
| `foldmanager.md` | Merge `folding.md` into `foldmanager.md`; absorb schedule tree field details |
| `foldproperty.md` | Move folding concept intro from `foldmanager.md` |

---

### Task — Per-object documentation quality pass (2026-08-14)

**Commits:** `cdd182c`, `5e77086`, `a49f1e8`, `1fde3b6`, `f885cdb`, `bb613ba`, `799b862`, `2c25a49`, `df95ec6`, `ddd5e57`, `fac2362`, `da80514`, `40632b2`, `b75de4a`, `ee7f76f`, `eba3f9b`, `ed4d18f`

| File | Change |
|---|---|
| `primarydsinfo.md`, `scheduletreenode.md` | Improve to documentation standard |
| `datastructdims.md`, `datastageparam.md` | Improve to documentation standard |
| `designspaceconfig.md` | Improve; remove Generated by / Written by annotations |
| `foldmanager.md` | Improve; update schema FoldManager descriptions |
| `foldproperty.md` | Improve; update schema FoldProperty descriptions |
| `superdsc-object.md` | Improve; reorder TOC; update nav chain; absorb `root-structure.md` and `wrappeddesignspaceconfig.md` |
| `constantinfo.md` | Improve; update schema ConstantInfo |
| `computeoperation.md` | Merge `operationattributes.md`; apply compliance findings F1–F13 |
| `memoryorganization.md` | Apply compliance findings F1–F8 |
| `labeleddatastructure.md` | Add intro and Context section; apply compliance findings F3–F13 |
| `coordinateinfo.md` | Improve with context, fields table, semantic descriptions, and example annotation |
| `coordinatecontainer.md` | Improve with context, fields table, and example |

---

### Task — Initial directory creation (2026-08-13)

**Commit:** `3317e79`

All API documentation files created for the first time under `sdsc_BUNDLE_APIs/`.
