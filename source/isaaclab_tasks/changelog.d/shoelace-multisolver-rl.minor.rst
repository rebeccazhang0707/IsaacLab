Added
^^^^^

* Added the ``IsaacContrib-Shoelace-DualFranka`` reinforcement learning example with standard
  ``ManagerBasedRLEnv``. Coupled MJWarp robot dynamics and VBD shoelace rods through ADMM.
* Included packaged USD assets and offline asset authoring, signed finger-tail contact observations,
  cooperative grasp-and-pull rewards and success criteria, settled startup poses, randomized resets,
  and an RSL-RL PPO configuration.
* Reused native material/collision configuration fragments and mass-derived cable inertia.
  Authored the remaining cable model overrides with ``NewtonCablePropertiesCfg`` before replication.
