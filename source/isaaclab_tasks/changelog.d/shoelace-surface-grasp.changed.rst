* **Breaking:** Unified Franka and Inspire shoelace acquisition, retention and success rewards
  around opposing surface geometry and worst-finger relative speed, replacing joint-angle closure
  assumptions. Added per-finger relative velocities while keeping actor and critic observations shared.
  Custom reward configurations now use ``finger_cfgs`` and environment surface/aperture parameters;
  see ``contrib/shoelace/GRASP_MDP.md`` for migration. Actor/critic inputs changed to 86/86 for
  dual-Franka and 112/112 for G1, so existing checkpoints require their original task revision.
  Removed the intermediate tail-direction and critic task-state terms and direction reward parameters.
