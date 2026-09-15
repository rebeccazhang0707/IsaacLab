Changed
^^^^^^^

* **Breaking:** Required bilateral loaded pulling before geometric completion in the dual-Franka shoelace
  task. Each tail accumulated at least 0.025 m of new outward records with valid grasps on both hands;
  passive movement, late closure, repeated excursions, and invalid observation gaps did not count.
  Completion of the cooperative phase persisted after release and cleared on episode reset.
  The success termination, success-rate metric, and +5 reward used the stricter result without changing
  reward weights or action and observation interfaces. Existing checkpoints required re-evaluation and
  running processes required restarting; earlier geometry-only success rates were not comparable.
  The original ``mdp.shoelace_success`` helper remained available for explicit legacy evaluation with
  only its geometric parameters.
* **Breaking:** Tightened default shoelace completion geometry from a total throat occupancy limit of 52
  capsule centers to at most 15 per arm's cable, implying at most 30 in total. Required each tail to reach its own signed
  outward X boundary, 0.09 m from the fixed midpoint, so one side could not compensate for the other.
  Earlier geometry-completion curves required re-evaluation. Custom tasks could configure the new
  ``maximum_throat_segments_per_arm`` and ``minimum_tail_outward_distance`` success parameters; explicit
  legacy evaluations retained the stateless helper with the saved run's original geometric parameters.
  Removed redundant radial-distance, total-X-separation, and total-count settings from the new cooperative
  success configuration; custom cooperative terms used only per-side geometric limits. Kept the legacy
  helper's original interface and derived the unchanged dense pull-normalization target from the per-side X goal.
* Consolidated grasp parameter defaults and removed redundant private helper layers without changing
  formulas or configuration values. Per-term parameter overrides remained independent; existing public
  configuration keys required no migration.

Added
^^^^^

* Added current geometry-completion and bilateral-pull-completion fractions and per-arm loaded-pull
  distance metrics. Exposed ``mdp.shoelace_grasp_quality`` as the single implementation of the contact,
  actual-closure, and low-slip grasp criterion for acquisition, retention, and success. Kept the formula
  unchanged and retained independent filters and episode histories for each term.
* Added per-arm throat capsule-count metrics and optional per-arm counts from ``untying_metrics`` while
  preserving its default total-count output and robot-arm ordering.
