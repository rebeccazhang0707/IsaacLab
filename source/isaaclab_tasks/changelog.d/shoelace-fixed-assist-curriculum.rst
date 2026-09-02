Changed
^^^^^^^

* Decoupled the eleven-level shoelace reset curriculum from grasp-assistance strength. Assistance was fixed for every
  level when enabled and could be disabled, including its spring latch, with ``grasp_assist_enabled=False`` or
  ``grasp_assist_maximum_force=0``. Disabled runs used geometric grasp state for grasp-loss termination.

Deprecated
^^^^^^^^^^

* Deprecated the ``approach_level_count`` and ``grasp_assist_strengths`` shoelace curriculum parameters. Existing
  configurations remained loadable, but users should configure eleven reset levels with ``level_count`` and select
  assisted or unassisted physics through ``grasp_assist_enabled``.
