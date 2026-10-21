* Limited robot collision geometry in the shoelace tasks to the Franka gripper fingers
  and the G1 Inspire thumb and index links, reducing Newton ADMM contact allocations.
  Preserved mass and inertia, shoe and support collisions, and cable
  self-collision. Existing policies require re-evaluation under the simplified contact
  model. Setting the Franka robots' ``spawn.collision_props`` to ``None`` restores their
  full authored collision geometry; G1 also uses a task-specific terminal-pad spawner.
