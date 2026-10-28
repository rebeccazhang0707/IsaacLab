* Limited full-rate shoelace grasp retention to two quality-weighted seconds per episode, then retained
  20 percent maintenance credit. Release and regrasp did not renew the budget. Observation, action, and
  success interfaces remained unchanged. Existing checkpoints require re-evaluation under the new objective.
  To restore the previous objective, set ``contact_penetration_tolerance=0.0`` in both dense and hold rewards,
  ``dense_task.params.pull_use_high_water_mark=False``, and ``grasp_hold.params.full_reward_duration=None``.
