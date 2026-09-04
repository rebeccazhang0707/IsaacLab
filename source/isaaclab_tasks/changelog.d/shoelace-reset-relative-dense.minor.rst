Changed
^^^^^^^

* Reworked the dual-Franka shoelace dense reward to use reset-relative potential differences, blend mean progress
  with the least-complete success margin, and activate untying and pull credit only after a confirmed bilateral
  strict grasp.
* Required simultaneous strict grasp confirmation for the shoelace acquisition event instead of allowing two
  sequential per-tail contacts to collect the full event reward. Initial reset grasps now seed the reward and
  termination latches without emitting an acquisition bonus, preventing discounted reset-and-drop reward farming.
  Custom reward configurations that require the previous absolute-state or per-tail acquisition behavior can keep
  using ``mdp.shoelace_dense_reward`` or ``mdp.grasp_acquisition_event`` directly.
* Increased the early-failure penalty so collecting the bilateral acquisition bonus and immediately losing a grasp
  no longer produced a positive net event return.
* Changed the pull-only curriculum reset to use a physically retained gripper aperture and added confirmation to
  geometric grasp release, preventing one-frame contact fluctuations from ending an episode. The early gripper
  reset levels were also made denser near the retained aperture to ease the first grasp-acquisition transition.
