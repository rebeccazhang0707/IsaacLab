* Added an independent ``pregrasp_progress_reward`` for fine shoelace TCP positioning and nearby actual
  gripper closure. Bounded, signed potential differences avoided repeated stationary credit, preserved
  state across invalid samples, and seeded per-environment resets without acquisition credit.
