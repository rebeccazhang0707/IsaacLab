Added
^^^^^

* Added a task-local fixed-observation-statistics RSL-RL model for stable shoelace checkpoint consolidation.

Changed
^^^^^^^

* Densified the dual-Franka shoelace reset curriculum across the micrometre-sensitive contact-release boundary
  and reduced the default environment spacing to preserve contact consistency at large environment counts. Also
  retuned the gripper drive to establish cable contact before an initially open tail could escape and calibrated
  promotion to achievable stochastic success windows.
  Configurations that resume a saved curriculum frontier should map it by the authored gripper reset position
  instead of relying on the former numeric level index.
