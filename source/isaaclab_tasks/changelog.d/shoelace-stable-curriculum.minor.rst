Added
^^^^^

* Added a 97-level calibrated dual-Franka shoelace reset curriculum that covered retained pulling, gripper release,
  contact acquisition, and staged arm approach states.
* Added task-local fixed-statistics and partially frozen RSL-RL actor variants for consolidating successful contact
  policies without changing their observation normalization.
* Added Gaussian arm exploration with Bernoulli binary-gripper decisions, avoiding Gaussian likelihood updates at
  the gripper sign boundary.

Changed
^^^^^^^

* Added exponential moving-average Cartesian arm commands and reset-dependent command warm-up. Existing policies
  that require immediate, unfiltered commands can set both arm-action ``alpha`` values to ``1.0`` and
  ``warmup_steps`` values to ``0``.
* Calibrated the gripper preload, ADMM interface solve, and curriculum reset density around measured Newton contact
  transitions. Checkpoint continuations should map historical levels by reset geometry because later numeric level
  indices shifted as bridge states were inserted.

Fixed
^^^^^

* Fixed deterministic hybrid-action export to use TorchScript-compatible slicing so the standard RSL-RL player
  could load the frozen curriculum checkpoint.
* Fixed reset-relative reward initialization, timeout accounting, exclusive failure rewards, and success-driven
  curriculum episode accounting so startup resets and finite-horizon episodes no longer supplied false credit.
* Added missed-acquisition, lost-grasp, and insufficient-separation state machines so stalled or released policies
  received a finite-horizon failure signal without treating transient contact motion as failure.
