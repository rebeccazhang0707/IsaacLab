* Corrected dual-Franka shoelace loaded-pull rewards by tolerating bounded contact-solver penetration
  and rewarding new grasp-qualified outward displacement records instead of changes in grasp-gated
  pull potential. Ungrasped records were consumed without credit to prevent retrospective regrasp rewards.
