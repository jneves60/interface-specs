# JSON Schema

[`sdscbundle-schema.json`](sdscbundle-schema.json) is the machine-readable contract for every
`sdsc_*.json` file in a SuperDSC-Bundle. It is written against
[JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12).

The schema enforces:

- Required fields and their types at every level of the object hierarchy
- Enum values for `dsType_`, `component_`, `indirectAllocType_`, `indexTensorType_`,
  `dataFormat_`, `fidelity_`, `nodeType_`, and all other constrained string fields
- Pattern constraints on dynamic keys such as core IDs and operation-name root keys
- Structural rules such as `additionalProperties: false` on all defined objects

Semantic constraints that go beyond what JSON Schema can express — cross-field consistency,
stick-size alignment, and symbol-ID correspondence — are documented in the individual object
pages and in the [Validation section of SDSC JSON API](SDSC-json-api.md#validation).

---

| [← Previous: Object Hierarchy](JSON-object-Hierarchy.md) | [↑ Table of Contents](README.md) | [Next: SuperDsc Object →](superdsc-object.md) |
|:--|:--:|--:|
