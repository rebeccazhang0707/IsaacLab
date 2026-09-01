Fixed
^^^^^

* Prevented non-finite shoelace state and finite joint-velocity outliers from destabilizing rewards during terminal
  simulation steps.
* Aligned the Franka task frame and closed-gripper target with the physical finger pads and shoelace diameter.
* Reset both Frankas around the authored shoelace tails with surface-oriented grippers instead of deforming the cable for the grasp curriculum.
* Added a geometry-gated, breakable compliant grasp to retain thin shoelace tails across one-way Newton proxy coupling.

Added
^^^^^

* Added a success-driven six-level curriculum that progresses from a stable start-in-hand pull to the complete
  authored approach-and-grasp reset while replaying the preceding level.
