Changed
^^^^^^^

* Replaced the authored dynamic cable reset with an offline gravity-settled, zero-velocity state, avoiding an episode
  warm-up transient.
* Recalibrated every dual-Franka shoelace curriculum reset level against the settled tail geometry, using mirrored
  near-normal wrist poses and measured discrete arm and gripper states instead of stale joint-space interpolation.
* Moved both near-tail grasp states inward along the free tails so the fingers maintain contact during closure and
  holding with ADMM coupling.
* Raised the baseline cable bend stiffness and damping, then applied a stronger smooth blend along the distal free
  tails to maintain tail clearance under gravity.
