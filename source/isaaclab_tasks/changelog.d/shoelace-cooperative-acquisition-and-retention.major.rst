Added
^^^^^

* Added an independent, time-scaled shoelace grasp-retention reward with a bilateral bonus and
  per-environment contact filters. Added a completion reward of five per geometric success event.
  Exposed ``grasp_hold_reward`` and ``shoelace_success_reward`` through the task MDP module.

Changed
^^^^^^^

* **Breaking:** Changed shoelace shaping to allocate 30% of progress potential to cooperative approach,
  30% to cooperative grasp acquisition, and 40% to pulling. Increased the approach width to 0.08 m
  and the bilateral pull fraction to 0.8, reducing single-hand initial-position pull credit.
  Changed ``approach_fraction`` to linearly divide acquisition between the two cooperative scores.
  Preserved parameter names and the observation/action interface, but historical reward curves and
  old parameter values no longer reproduced the previous objective. Re-evaluate or retrain policies,
  set ``bilateral_approach_fraction`` and ``bilateral_grasp_fraction`` explicitly in custom reward
  configurations, and disable ``grasp_hold`` or ``success`` by weight for independent ablations.
