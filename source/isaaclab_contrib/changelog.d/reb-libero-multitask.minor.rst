Added
^^^^^

* Added a trainable LIBERO-Spatial / LIBERO-Goal multi-task manipulation
  environment (``Isaac-Libero-Spatial-Goal-Franka-Multi-Task-v0`` and its
  ``-Play-v0`` variant) built on the
  :class:`~isaaclab_contrib.tasks.manipulation.multitask.registry.MultiTaskRegistry`,
  with a shared Franka robot module, per-suite task modules, LIBERO-specific MDP
  rewards, and an RSL-RL PPO agent config.
* Added LIBERO-Object and LIBERO-Long multi-task environments
  (``Isaac-Libero-Object-Franka-Multi-Task-v0``,
  ``Isaac-Libero-Long-Franka-Multi-Task-v0`` and their ``-Play-v0`` variants),
  whose per-task object layouts are loaded from the LIBERO per-task JSON config
  (normalised to a shared Franka base frame) and built through the same
  object-level prototype-sharing factory.
* Added a combined multi-suite LIBERO environment
  (``Isaac-Libero-All-Franka-Multi-Task-v0`` for all 40 tasks, plus a
  representative cross-suite ``Isaac-Libero-Object-Long-Franka-Multi-Task-v0`` and
  their ``-Play-v0`` variants) that trains any subset of the four suites together
  with object-level prototype sharing: identical object models are harvested and
  de-duplicated so each distinct USD is spawned once and cloned -- as a single
  ``AssetView`` -- into the union of the envs whose task uses it. A deterministic
  ``sequential`` clone map assigns env ``i`` to task ``i % n_tasks``, driving
  per-env ``task_onehot``, object observations, the three reward modes, the
  success termination, and per-task object reset. Arbitrary suite combinations can
  be built with
  :func:`~isaaclab_contrib.tasks.manipulation.libero.config.franka.libero_all_env_cfg.make_libero_combined_env_cfg`.
* Added single-suite harvest environments ``Isaac-Libero-Goal-Franka-Multi-Task-v0``
  and ``Isaac-Libero-Spatial-Franka-Multi-Task-v0`` (and their ``-Play-v0``
  variants), each training one suite's 10 tasks through the same object-level
  prototype-sharing factory. The goal-only env in particular handles
  ``libero_goal`` tasks whose relational / articulation goals have no manipulated
  rigid object (no manipulated-object reset, reward, or drop termination is
  applied for those tasks).
* Fixed the ``libero_object`` ``floor`` fixture, whose 6 m x 6 m footprint spilled
  into neighbouring environments; its XY footprint is now shrunk to ~2.1 m (Z left
  unchanged so object rest heights are preserved) to fit within the env spacing.
* Added three selectable LIBERO reward modes (``goal``, ``metaworld_dense``,
  ``world_prediction``), applied uniformly across all four suites and selectable
  via the ``LIBERO_REWARD_MODE`` environment variable or the task cfg
  ``reward_mode`` field (default ``goal``): a sparse goal-success bonus with a
  matching success termination, a demo-free dense reach/lift/place shaping, and a
  gated dense demo-tracking mode (falls back to the sparse bonus when no per-step
  demo reference is available).
