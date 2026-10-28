* Added current geometry-completion and bilateral-pull-completion fractions and per-arm loaded-pull
  distance metrics. Exposed ``mdp.shoelace_grasp_quality`` as the single implementation of the contact,
  actual-closure, and low-slip grasp criterion for acquisition, retention, and success. Kept the formula
  unchanged and retained independent filters and episode histories for each term.
* Added per-arm throat capsule-count metrics and optional per-arm counts from ``untying_metrics`` while
  preserving its default total-count output and robot-arm ordering.
