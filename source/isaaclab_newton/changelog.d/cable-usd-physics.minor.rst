Added
^^^^^

* Added optional asset-local USD cable physics attributes for fixed segments, inertia,
  initial orientations, per-joint stiffness and damping, and VBD Dahl parameters. Applied
  them before source-asset replication in the physics and visualization import paths.
* Added ``NewtonCablePropertiesCfg`` through the existing ``SchemaFragment`` interface
  for cable overrides unavailable in native USD import. Reused standard material and
  collision configurations, and used native segment masses to derive geometric inertia.
* Added ``isaaclab:physics:fixed`` for explicit zero-mass rigid fixtures while preserving
  their imported body and joint topology.

Fixed
^^^^^

* Transferred authored rigid-contact friction, stiffness, damping, and contact gaps to
  imported cable capsule shapes.
