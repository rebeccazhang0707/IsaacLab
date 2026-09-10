Changed
^^^^^^^

* Added small action-change and action-magnitude penalties to the dual-Franka shoelace task, selecting only
  normalized arm commands and excluding binary gripper actions. Set
  ``env.rewards.arm_action_rate.weight=0`` and ``env.rewards.arm_action_magnitude.weight=0`` to restore the
  previous reward objective.

Added
^^^^^

* Added per-step approach, grasp, pull, validity, and recent-episode success metrics under
  ``Metrics/shoelace/`` for RSL-RL terminal output and TensorBoard logging.
