* **Breaking:** Renamed the shoelace policy observation ``gripper_close_error`` to ``finger_tracking_error``
  without changing its values, dimensions or concatenation order. Renamed its task helpers to
  ``gripper_closing_tracking_error`` and ``inspire_finger_tracking_error``; update custom imports
  and named observation references. Franka retained closing-direction-only errors, while G1
  retained signed errors for all four thumb/index joints.
