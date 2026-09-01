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

* Added a success-driven eleven-level curriculum that first opened the grippers at the authored tails, then increased
  approach distance toward the complete task with finer early approach levels, transition-balanced replay strata,
  gradual frontier exposure, and low-success exposure backoff.

* Restricted compliant grasp assistance to cable tails within 18 mm of a gripper TCP and inside its closing finger
  opening, released it at 35 mm, ramped its force from near zero as the fingers closed, and retained its 2 N maximum
  force.

* Added a per-tail reward for acquiring the cable within the grasp-assistance distance.
