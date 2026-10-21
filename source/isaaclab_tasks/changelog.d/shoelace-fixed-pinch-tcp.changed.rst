* **Breaking:** Replaced G1's live-midpoint IK and finger-motion feedforward with standard
  differential IK at calibrated nominal closed-pinch wrist offsets. G1 coarse approach and
  TCP observations used that same fixed reference throughout opening and closing, without
  handoff; live finger geometry, contact, and slip still determined physical grasp quality.
  Removed ``InspirePinchIKAction`` and ``InspirePinchIKActionCfg``; custom configurations must
  use ``DifferentialInverseKinematicsActionCfg`` with a calibrated ``body_offset`` and bind
  ``approach_robot_cfgs`` to the wrist bodies when selecting TCP-based coarse approach.
  Existing G1 policies require retraining despite unchanged action and observation dimensions.
  Dual-Franka retained its capture-distance approach reward.
* Changed G1 binary gripper actions to move each index finger while holding the thumb at
  its calibrated closed-pinch target.
* **Breaking:** Made straight fingers and invisible 16 x 10 x 3 mm terminal collision pads
  the G1 defaults. Removed six downstream thumb/index mimic constraints and restricted those
  joints near zero; recalibrated targets, surface references and wrist TCPs for a 2.8 mm gap.
  The original USD remained unchanged, but the simulated hand no longer reproduced its mechanical
  coupling. Existing coupled-hand checkpoints require retraining under the new mechanism.
