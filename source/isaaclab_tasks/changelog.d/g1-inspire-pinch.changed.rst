* **Breaking:** Replaced whole-hand G1 Inspire shoelace synergies with synchronous thumb/index
  open/close commands, kept the remaining fingers straight, and added calibrated finger targets,
  mimic-consistent ramps, and finger observations. The live finger-surface midpoint defined the TCP
  for arm IK, approach rewards and slip measurements. The task used 14 actions and 102 policy
  observations; existing 14-action, 64-observation checkpoints require the previous
  task revision for playback and cannot resume training with this configuration.
