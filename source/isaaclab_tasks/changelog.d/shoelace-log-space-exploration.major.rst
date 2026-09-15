Changed
^^^^^^^

* **Breaking:** Changed the shoelace PPO actor's default Gaussian standard-deviation parameterization
  from scalar to log space while retaining an initial standard deviation of 1.0 and the same action
  interface. Scalar and log configurations use different checkpoint parameter names. Load scalar-trained
  checkpoints with their saved agent configuration or set ``agent.actor.distribution_cfg.std_type=scalar``.
  Corrected the rollout documentation to match the existing eight-step default.
