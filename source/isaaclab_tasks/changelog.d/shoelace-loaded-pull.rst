Fixed
^^^^^

* Corrected dual-Franka shoelace loaded-pull rewards by tolerating bounded contact-solver penetration
  and rewarding new grasp-qualified outward displacement records instead of changes in grasp-gated
  pull potential. Ungrasped records were consumed without credit to prevent retrospective regrasp rewards.

Changed
^^^^^^^

* Limited full-rate shoelace grasp retention to two quality-weighted seconds per episode, then retained
  20 percent maintenance credit. Release and regrasp did not renew the budget. Observation, action, and
  success interfaces remained unchanged. Existing checkpoints require re-evaluation under the new objective.
  To restore the previous objective, set ``contact_penetration_tolerance=0.0`` in both dense and hold rewards,
  ``dense_task.params.pull_use_high_water_mark=False``, and ``grasp_hold.params.full_reward_duration=None``.
