# SDSC Bundle API Documentation

---

## Table of Contents

Documents are organized in the recommended reading order, grouped by learning stage.

---

### Stage 1 — Orientation (What & Why)

> Understand the Spyre stack, the two API components (MLIR + JSON), and the overall JSON file structure before reading individual object specs.

| # | Section | File |
|---|---------|------|
| 1.1 | Overview | [Overview.md](Overview.md) |
| 1.2 | SDSC JSON API | [SDSC-json-api.md](SDSC-json-api.md) |
| 1.3 | Object Hierarchy | [JSON-object-Hierarchy.md](JSON-object-Hierarchy.md) |

---

### Stage 2 — Core Concepts (Foundational Objects)

> Folding, padding, and stick layout are foundational; every later object references these ideas.

| # | Section | File |
|---|---------|------|
| 2.1 | SuperDsc Object | [superdsc-object.md](superdsc-object.md) |
| 2.2 | FoldProperty | [foldproperty.md](foldproperty.md) |
| 2.3 | FoldManager | [foldmanager.md](foldmanager.md) |
| 2.4 | Padding | [padding.md](padding.md) |
| 2.5 | Stick Layout Constraints | [stick-layout-constraints.md](stick-layout-constraints.md) |

---

### Stage 3 — DesignSpaceConfig & Tensor Objects

> The `dscs_[]` array and the tensor-descriptor objects that make up the bulk of every real SDSC JSON file.

| # | Section | File |
|---|---------|------|
| 3.1 | DesignSpaceConfig | [designspaceconfig.md](designspaceconfig.md) |
| 3.2 | LabeledDataStructure | [labeleddatastructure.md](labeleddatastructure.md) |
| 3.3 | MemoryOrganization | [memoryorganization.md](memoryorganization.md) |
| 3.4 | PrimaryDsInfo | [primarydsinfo.md](primarydsinfo.md) |
| 3.5 | DataStructDims | [datastructdims.md](datastructdims.md) |

---

### Stage 4 — Execution & Scheduling Objects

> Per-core data staging, the schedule tree, coordinate mappings, compute operations, and constant tensors.

| # | Section | File |
|---|---------|------|
| 4.1 | DataStageParam | [datastageparam.md](datastageparam.md) |
| 4.2 | ScheduleTreeNode | [scheduletreenode.md](scheduletreenode.md) |
| 4.3 | CoordinateContainer | [coordinatecontainer.md](coordinatecontainer.md) |
| 4.4 | CoordinateInfo | [coordinateinfo.md](coordinateinfo.md) |
| 4.5 | ComputeOperation | [computeoperation.md](computeoperation.md) |
| 4.6 | ConstantInfo | [constantinfo.md](constantinfo.md) |

---

### Stage 5 — Worked Examples

> End-to-end examples that tie all objects together; read after completing Stage 4.

| # | Section | File |
|---|---------|------|
| 5.1 | Complete Example (JSON) | [complete-example.md](complete-example.md) |

---

### Stage 6 — MLIR Bundle (Orchestration Layer)

> The `.mlir` file that orchestrates execution flow, symbol management, and operation sequencing across one or more SDSC JSON files.

| # | Section | File |
|---|---------|------|
| 6.1 | MLIR Bundle API | [MLIR-bundle-API.md](MLIR-bundle-API.md) |
| 6.2 | Bundle Usage Examples | [MLIR-bundle-usage-examples.md](MLIR-bundle-usage-examples.md) |
| 6.3 | Complete Example (MLIR) | [MLIR-complete-example.md](MLIR-complete-example.md) |

---

### Stage 7 — Production Readiness

> Error handling, authoring best practices, and the reference glossary.

| # | Section | File |
|---|---------|------|
| 7.1 | Error Handling | [error-handling.md](error-handling.md) |
| 7.2 | Best Practices | [best-practices.md](best-practices.md) |
| 7.3 | Reference Documents & Glossary | [reference.md](reference.md) |

---
