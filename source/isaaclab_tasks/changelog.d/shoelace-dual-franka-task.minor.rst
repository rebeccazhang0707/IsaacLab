Added
^^^^^

* Added ``IsaacContrib-Shoelace-DualFranka``, a manager-based Newton task that trains two Franka
  robots to approach, grasp, and untie shoelace tails through coupled MJWarp/VBD contact, with
  graph-safe VBD contact-history configuration, headless-safe shoe collider authoring, and
  training-safe termination, gravity-compensated arm control, root-frame task observations, cable-safe gripper
  targets, dense approach and grasp shaping, checkpoint settings, and a validated lower-cost 360-segment,
  four-substep, 16-iteration cable configuration. The cable used two dynamic end chains with fixed seam anchors,
  a visual-only middle span, and short static seam-contact meshes to avoid simulating the eyelet-held interior. The
  policy observation omitted constant pull directions, algebraically redundant tail separation, episode time, and
  inferred grasp state, while throat density moved to the critic-only privileged observation group.
