Changed
^^^^^^^

* Moved the shoelace task's non-Dahl model overrides from USD authoring to the
  ``configure_shoelace_physics`` startup event. Updated fixed-segment and shoe mass/inertia,
  geometric cable inertia plus ``cable_inertia_regularization``, and graded tail stiffness/damping
  through Newton runtime arrays and the existing coupler model-change notification path.
* Kept settled positions, orientations, and zero velocities in ``install_settled_default_state``;
  episode resets retained state restoration and randomization without changing physics parameters.
* Moved cable contact friction, stiffness/damping, and gap configuration into the same startup
  event, with ``SHAPE_PROPERTIES`` notifications for compact solver views. Kept finger/shoe
  spawner friction overrides and construction-time Dahl USD import.
* **Breaking:** Rejected the deprecated ``ShoelaceUsdCfg.inertia_regularization`` setting.
  Use the startup event and task-level ``cable_inertia_regularization`` instead of baking
  non-Dahl overrides into USD.
