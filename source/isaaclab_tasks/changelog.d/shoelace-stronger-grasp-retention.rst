Changed
^^^^^^^

* Increased the shoelace grasp-retention weight from 0.2 to 1.0 and reduced its bilateral fraction from
  0.8 to 0.5, raising ideal single-arm retention credit from 0.02 to 0.25 per second and bilateral credit
  from 0.2 to 1.0 per second. Kept the grasp criteria unchanged and updated the budget tests and docs.
  To restore the previous objective, set ``env.rewards.grasp_hold.weight=0.2`` and
  ``env.rewards.grasp_hold.params.bilateral_grasp_fraction=0.8``. Re-evaluate completion alongside
  retention duration because stationary holding can compete with pulling.
