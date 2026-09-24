Changed
^^^^^^^

* Removed Dahl parameter authoring from the shoelace task and regenerated its USD
  asset without custom cable attributes. Retained task-specific physics in the
  existing startup event and loaded the asset through native Newton USD import.
* Selected compliant ALM through the existing ``VBDSolverCfg.rigid_compliant_alm``
  option and reused the completed VBD solver configuration dependency without a
  task-local solver configuration subclass.
