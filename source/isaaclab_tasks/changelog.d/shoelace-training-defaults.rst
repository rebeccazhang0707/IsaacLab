Changed
^^^^^^^

* Raised the dual-Franka shoelace PPO initial action standard deviation from 0.3 to 1.0 and reduced VBD
  solver iterations from 20 to 12 in the task and standalone Franka demo. Restore ``init_std=0.3`` in
  the agent configuration and ``VBD_ITERATIONS=20`` in the task and demo to recover the previous defaults.
  Resumed checkpoints retained their saved action standard deviations; re-evaluate contact behavior when
  comparing runs with different solver iteration counts.
