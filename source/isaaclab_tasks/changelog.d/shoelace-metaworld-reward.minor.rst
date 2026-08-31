Changed
^^^^^^^

* Replaced independent shoelace approach, grasp, progress, and directional shaping terms with a bounded
  Meta-World-style dense reward that composes dual-tail acquisition and untying goals using Hamacher soft-AND.
* Expressed success and early-failure rewards as direct per-event bonuses, with a 60-point success bonus and a
  2-point penalty for each non-timeout failure reason.
* Synchronized the shoelace anchors, seam-local collision meshes, tongue collider, and Newton solver cadence with
  the validated standalone demo.
