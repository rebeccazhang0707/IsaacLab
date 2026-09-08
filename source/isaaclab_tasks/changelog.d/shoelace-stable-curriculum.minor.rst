Added
^^^^^

* Added a 97-level calibrated dual-Franka shoelace reset curriculum that covered retained pulling, gripper release,
  contact acquisition, and staged arm approach states.
* Added task-local fixed-statistics and partially frozen RSL-RL actor variants for consolidating successful contact
  policies without changing their observation normalization.
* Added Gaussian arm exploration with Bernoulli binary-gripper decisions, avoiding Gaussian likelihood updates at
  the gripper sign boundary.
* Added a signed hand-frame contact-socket observation, checkpoint interface converter, and behavior-cloning tools
  for socket-aware BC and DAgger consolidation.
* Added a binary gripper target-rate limiter so contact entry can be slowed without reducing PPO arm exploration.

Changed
^^^^^^^

* Added exponential moving-average Cartesian arm commands and reset-dependent command warm-up. Existing policies
  that require immediate, unfiltered commands can set both arm-action ``alpha`` values to ``1.0`` and
  ``warmup_steps`` values to ``0``.
* Calibrated the gripper preload, ADMM interface solve, and curriculum reset density around measured Newton contact
  transitions. Checkpoint continuations should map historical levels by reset geometry because later numeric level
  indices shifted as bridge states were inserted.
* Changed the actor and critic interfaces from 62/63 to 68/69 values by inserting six socket-error values. Existing
  62-value checkpoints must first be converted with ``scripts/demos/shoelace_expand_socket_observation.py``.
* Required the calibrated contact socket for acquisition, retention, and task shaping, and extended the acquisition
  deadline to cover the deliberately slower gripper target motion.

Fixed
^^^^^

* Fixed deterministic hybrid-action export to use TorchScript-compatible slicing so the standard RSL-RL player
  could load the frozen curriculum checkpoint.
* Fixed reset-relative reward initialization, timeout accounting, exclusive failure rewards, and success-driven
  curriculum episode accounting so startup resets and finite-horizon episodes no longer supplied false credit.
* Added missed-acquisition, lost-grasp, and insufficient-separation state machines so stalled or released policies
  received a finite-horizon failure signal without treating transient contact motion as failure.
