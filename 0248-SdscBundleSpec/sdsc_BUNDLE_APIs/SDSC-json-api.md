# SDSC JSON API

## Introduction

Each `sdsc_*.json` file in a SuperDSC-Bundle describes a single torch operation to be executed on the Spyre backend (DeepTools). It is a self-contained compiled artifact — one JSON file encodes everything the hardware needs to execute that operation deterministically across all 32 cores: how the iteration space is divided, how tensors are laid out in memory, where data lives (HBM vs. LX scratchpad), and what compute to perform.

A PyTorch model may translate into several SuperDSC-Bundles. Each bundle is composed of a `bundle.mlir` file (which orchestrates execution flow and symbol management — see [MLIR Bundle API](MLIR-bundle-API.md)) and one or more `sdsc_*.json` files. Each JSON file corresponds to one torch operation.

The different sections of an SDSC JSON file describe the following:

- **Core fold properties** — how the iteration space is divided across cores
- **Tensor descriptors** — layout, memory residency, data format, and stick configuration for each tensor
- **Schedule tree** — per-tensor memory allocation, start addresses, and coordinate mappings
- **Data staging** — per-core tile sizes for steady-state and epilogue passes
- **Compute operations** — execution unit, operation name, and input/output tensor references

## Key Components

An SDSC JSON file is structured as a single top-level key (the operation name) whose value is a **SuperDsc** object. The SuperDsc object holds a few top-level fields and a `dscs_[]` array of **DesignSpaceConfig** entries. Each `dscs_[]` entry is itself a single-key object wrapping a `DesignSpaceConfig` — some of its fields are leaf values while others are composite objects that drill down further into the object hierarchy.

| Component | Role | Reference |
|---|---|---|
| **SuperDsc** | Root object — holds fold properties, work-slice maps, core schedule, and the `dscs_[]` array | [superdsc-object.md](superdsc-object.md) |
| **DesignSpaceConfig** | Per-operation configuration — tensors, staging, schedule, and compute | [designspaceconfig.md](designspaceconfig.md) |
| **LabeledDataStructure** | Tensor descriptor — role, format, scale, and memory residency | [labeleddatastructure.md](labeleddatastructure.md) |
| **PrimaryDsInfo** | Tensor-type layout — memory dimension order and stick configuration | [primarydsinfo.md](primarydsinfo.md) |
| **DataStageParam** | Per-core tile sizes for steady-state and epilogue passes | [datastageparam.md](datastageparam.md) |
| **ScheduleTreeNode** | Memory allocation node — component, start addresses, coordinates | [scheduletreenode.md](scheduletreenode.md) |
| **ComputeOperation** | Compute specification — execution unit, op name, tensor references | [computeoperation.md](computeoperation.md) |
| **MemoryOrganization** | HBM / LX scratchpad residency flags per tensor | [memoryorganization.md](memoryorganization.md) |

## Object Hierarchy

See [JSON-object-Hierarchy.md](JSON-object-Hierarchy.md) for the full hierarchy diagram.

## SuperDsc Object

See [superdsc-object.md](superdsc-object.md) for the complete field reference.

## FoldProperty

See [foldproperty.md](foldproperty.md) for details.

## FoldManager

See [foldmanager.md](foldmanager.md) for details.

## SuperDSC Object Fields

The SuperDSC object is the root container that encapsulates all information needed to execute an operation on the Spyre backend.

### Core Metadata and Folding information

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `<_name>`, e.g., `"0_batchmatmul"` | Identifier for the sdsc | |
| 2 | `numCoresUsed_` | Number of cores used for the operation | |
| 3 | `sdscFoldProps_` | Denotes folds over time | Vector of length 1 containing fields **factor_** and **label_**, set to 1 and "time", resp., as in `[{"factor":1, "label_": "time"}]`. |
| 4 | `sdscFolds_` | Contains sub-fields **dim_prop_func** and **dim_prop_attr**, describing folds over time. | Refer to default values in sample sdscs, which are sufficient in most of the cases. |
| 5 | `coreFoldProp_` | Denotes folding over cores using sub-fields **factor** and **label_** | factor=number of cores used and label='core' |
| 6 | `coreletFoldProp_` | Denotes folding over corelets using sub-fields **factor** and **label_** | factor=1 or 2 and label='corelet' |
| 7 | `fold_coord_` | For a folded SDSC with multiple fold types, fold_coord_ stores the index at each fold dimension. Specifies which unfolded variant an SDSC represents by storing its coordinate indices in folded space. | |

### Dimensions and Work Slicing

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 8 | `N_` | List of all dimensions across tensors used by ops in all dsc_'s in the sdsc. Contains each dimension's size across all cores. For dimensions that are padded (such as convolution's image dimensions) padding details are also included. See [Padding](padding.md). | |
| 9 | `numWkSlicesPerDim_` | A map keyed by the dimension name indicating the number of slices into which each dimension is split. | The product of the slices over all dimensions should equal the number of cores (across which the operation is executed). |
| 10 | `coreIdToWkSlice_` | Map from core id to the slice of each dimension assigned to it. | Nested map. Outer key is core id. Inner key is dimension name. The slice number assigned to a core ranges from 0 to the total slice count for the dim indicated by `numWkSlicesPerDim_`. |
| 11 | `coreIdToDsc_` | Mapping from core id to dsc number when the sdsc contains multiple dsc's. | |
| 12 | `ldsShareInfo` | LabeledDS sharing information. Tracks tensor sharing across multiple DSC instances. | |
| 13 | `opFuncsUsed_` | | |
| 14 | `prodConsList_` | | |

### Scheduling Related

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 15 | `coreIdToDscSchedule_` | Vector of DscScheduleStep schedule steps specified for each core. Defines the execution sequence of operations on each core by specifying which data and dl dsc operations execute on this core, the order they execute in, and data-flow dependencies. | |
| 16 | `target_` | Specifies the target backend/hardware platform for executing the SDSC op. | Can be one of SENTIENT, SENULATOR, SENPCFG, SENTF, SYSTEMC, R5SS, HOST |

### Nested Data Structures

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 17 | `dscs_` | Array of DesignSpaceConfig (or DataStageConfig) structures. | Vector to express core work mapping for an operation. With balanced work division, only one entry in the vector is needed. |

## DesignSpaceConfig Fields

See [DesignSpaceConfig](designspaceconfig.md) for the complete field reference.

## Tensor Description (LabeledDSInfo) Fields

See [LabeledDataStructure](labeleddatastructure.md) for the complete field reference, including the full list of supported `dataFormat_` values.

## ScheduleTree Node Fields

See [ScheduleTreeNode](scheduletreenode.md) for the complete field reference.

## Compute Operation Fields

See [ComputeOperation](computeoperation.md) for the complete field reference.

---

| [← Previous: MLIR Complete Example](MLIR-complete-example.md) | [↑ Table of Contents](README.md) | [Next: Object Hierarchy →](JSON-object-Hierarchy.md) |
|:--|:--:|--:|
