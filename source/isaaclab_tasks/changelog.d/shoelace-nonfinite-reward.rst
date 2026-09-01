Fixed
^^^^^

* Prevented non-finite shoelace state and finite joint-velocity outliers from destabilizing rewards during terminal
  simulation steps.
* Aligned the Franka task frame and closed-gripper target with the physical finger pads and shoelace diameter.
* Reset both Frankas around the authored shoelace tails with surface-oriented grippers instead of deforming the cable for the grasp curriculum.
* Added a geometry-gated, breakable compliant grasp to retain thin shoelace tails across one-way Newton proxy coupling.
* Aligned grasp-loss termination with the compliant grasp latch and its geometric release distance.
* Increased the per-body VBD rigid-contact capacity to prevent contact truncation in dense shoe and gripper states.

Added
^^^^^

* Added a success-driven fifteen-level curriculum that first staged gripper closure at the authored tails, then
  increased approach distance toward the complete task with finer early approach levels, transition-balanced replay
  strata, gradual frontier exposure, low-success exposure backoff, and four final levels that annealed compliant grasp
  assistance to zero.
