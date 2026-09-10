Changed
^^^^^^^

* Made the dual-Franka shoelace acquisition potential reward each hand's grasp independently of TCP proximity.
  Grasping either tail earned one share of the grasp budget, and grasping both earned two shares. Preserved
  no-grasp approach credit, the requirement for both grasps in the pulling score, and signed penalties for
  losing grasp. Existing configurations and checkpoints remained loadable; re-evaluate or retrain policies
  with the changed reward objective and compare grasp quality and success instead of old and new returns.
