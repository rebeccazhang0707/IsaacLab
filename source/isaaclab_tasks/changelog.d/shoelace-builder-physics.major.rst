Changed
^^^^^^^

* Moved static shoelace physics configuration to task-local construction before solver
  creation, including cable contact materials, inertia regularization,
  anchor and shoe mass/inertia, and graded joint stiffness/damping. Retained Newton's native
  finalize-time inertia validation instead of restoring sub-threshold inertias afterward;
  zero or very small regularization may therefore produce different inertias than before.
  Kept settled-state installation
  and episode randomization as state-only events. Construct the task with ``create_shoelace_env``
  or the registered Gym environment and set ``ShoelaceEnvCfg.cable_inertia_regularization``
  before construction.

* Removed the task's reference-orientation override and retained Newton's native USD-imported
  body orientations. Kept settled episode poses unchanged. Re-evaluate existing policies
  because the cable's material reference state may differ.

Removed
^^^^^^^

* **Breaking:** Removed the old ``configure_shoelace_physics`` startup event and the deprecated
  ``ShoelaceUsdCfg.inertia_regularization`` field. Remove the startup event from custom
  configurations and replace direct ``ManagerBasedRLEnv`` construction with ``create_shoelace_env``
  or the registered Gym environment to apply the task's static physics settings.
