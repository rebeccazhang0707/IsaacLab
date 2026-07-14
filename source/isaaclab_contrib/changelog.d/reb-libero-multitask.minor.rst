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
  per-env success termination and per-task object reset. Arbitrary suite
  combinations are built with
  :func:`~isaaclab_contrib.tasks.manipulation.libero.envs.dgpo_env_cfg.make_libero_dgpo_env_cfg`.
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
* Added a DGPO training path on the harvest / cloner API
  (:func:`~isaaclab_contrib.tasks.manipulation.libero.envs.dgpo_env_cfg.make_libero_dgpo_env_cfg`,
  layout :mod:`~isaaclab_contrib.tasks.manipulation.libero.dgpo_layout`)
  that locks actor/critic/action dims (324 / 572 / 7), provides a first-class
  OSC Franka robot module
  (:class:`~isaaclab_contrib.tasks.manipulation.libero.robots.franka_osc.LiberoFrankaOscRobotCfg`),
  and DGPO observation groups (real multi-hot, ``ObjectTargetPoseBuffer`` over
  shared prototypes, proprio + contact force) for gym ids
  ``Isaac-Libero-All-Dgpo-Osc-v0`` / ``Isaac-Libero-All-Dgpo-Osc-Play-v0``
  (plus single-suite / combo ``*-Dgpo-Osc-*`` ids; Compat remains a thin All
  alias; primary cfg
  :class:`~isaaclab_contrib.tasks.manipulation.libero.config.franka.libero_dgpo_env_cfg.LiberoAllDgpoOscEnvCfg`).
* Added demo / policy quaternion-order flags
  (``policy_quat_order`` / ``demo_quat_order``, default ``wxyz``) via
  :mod:`~isaaclab_contrib.tasks.manipulation.libero.mdp.quat` at load and obs
  boundaries (sim math stays XYZW).
* Added a demo command stack
  (:class:`~isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands.SourceLiberoCommand`
  + semantic command manager) so privileged EE/joint/object pose diffs become
  real when ``LIBERO_ASSEMBLED_DATASET_DIR`` demos are present (zeros kept
  otherwise with locked widths).
* Added demo-conditioned reset
  (:func:`~isaaclab_contrib.tasks.manipulation.libero.mdp.demos.events.reset_libero_scene_to_demo_initial_state`)
  that restores objects / articulations / robot joints from the reserved demo
  ``initial_state`` (aligned with :class:`~isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands.SourceLiberoCommand`
  via ``reserve_trajectories_for_envs``), using harvest
  ``filter_reset_ids`` / prototype name mapping.
* Extended ``scripts/benchmarks/profile_libero_scene.py`` with
  ``--impl compat_osc`` for build/step profiling of the DGPO OSC path.
* Added :class:`~isaaclab_contrib.tasks.manipulation.libero.config.franka.agents.rsl_rl_ppo_cfg.LiberoAllDgpoPPORunnerCfg`
  (``[512, 256, 128]`` MLPs, ``noise_std_type="log"``; ``LiberoAllCompatPPORunnerCfg``
  alias) for ``Isaac-Libero-*-Dgpo-Osc(-Play)-v0`` so DGPO training and legacy
  checkpoints share the same obs layout under rsl-rl >= 4.0.

Changed
^^^^^^^

* Promoted the DGPO-on-harvest stack out of ``compat/`` into first-class modules:
  :mod:`~isaaclab_contrib.tasks.manipulation.libero.dgpo_layout`,
  :mod:`~isaaclab_contrib.tasks.manipulation.libero.mdp.observations`,
  :mod:`~isaaclab_contrib.tasks.manipulation.libero.mdp.demos`, and
  :mod:`~isaaclab_contrib.tasks.manipulation.libero.envs`.
* Simplified Franka gym registration to the DGPO OSC path only
  (``Isaac-Libero-*-Dgpo-Osc(-Play)-v0``; All Compat ids kept as thin aliases).
  Removed the Diff-IK ``*-Franka-Multi-Task-*`` factory
  (:mod:`~isaaclab_contrib.tasks.manipulation.libero.config.franka.libero_all_env_cfg`),
  its Diff-IK Franka robot module, and the orphaned Diff-IK RSL-RL runner cfgs.
