Fixed
^^^^^

* Scaled Shoelace's internal ADMM contact-reduction hashtable with the per-process environment
  count to avoid contact-reduction hashtable saturation in large training batches.
  Preserved contact matching, its deterministic triangle-pair capacity limit, and
  larger explicit budgets and hashtable factors.

Changed
^^^^^^^

* Reduced Shoelace's RSL-RL rollout length from 32 to 16 steps per environment.
  Set ``agent.num_steps_per_env=32`` to restore the previous batch size and rollout length.
