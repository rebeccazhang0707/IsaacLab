* Enabled ``pregrasp`` shaping with a 15 mm alignment width and a smooth closure gate inside 10 mm.
  Preserved the existing task-progress, physical grasp-retention, success, action, and observation contracts.
  Set ``env.rewards.pregrasp.weight=0`` to restore the previous objective, or disable either component with
  ``alignment_weight=0`` or ``closure_weight=0`` in its parameters for focused ablations. Re-evaluate
  existing checkpoints before comparing reward curves; the positional gate was not a physical grasp test.
