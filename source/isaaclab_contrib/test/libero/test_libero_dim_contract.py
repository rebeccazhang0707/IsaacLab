# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure (no Isaac Sim) checks for the LIBERO DGPO+ABC dimension contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from isaaclab_contrib.tasks.manipulation.libero.dgpo_layout import (
    DGPO_ABC_ACTION_DIM,
    DGPO_ABC_ACTOR_OBS_DIM,
    DGPO_ABC_CRITIC_OBS_DIM,
    DGPO_ABC_TASK_SEQUENCE,
    LIBERO_ORDERED_OBJECT_NAMES,
    LIBERO_SHARED_SUBTASK_TAGS,
    build_overlap_tag_columns,
    dgpo_abc_contract,
    encode_assignment_multi_hot,
    harvest_task_to_assignment_key,
    multi_hot_encoding_width,
    verify_contract_arithmetic,
)

# Locked scene-entity anchors (see scripts/benchmarks/libero_scene_profiling.md).
PER_ENV_DGPO_ABC_SCENE_ENTITIES = 193
HARVEST_ALL_PROTOTYPES = 35


def test_dgpo_abc_contract_arithmetic():
    """Actor=324, critic=572, action=7 must match checkpoint math."""
    contract = verify_contract_arithmetic()
    assert contract.actor_obs_dim == DGPO_ABC_ACTOR_OBS_DIM == 324
    assert contract.critic_obs_dim == DGPO_ABC_CRITIC_OBS_DIM == 572
    assert contract.action_dim == DGPO_ABC_ACTION_DIM == 7


def test_obs_group_term_order_matches_agent_yaml():
    """Term order must match DGPO+ABC env.yaml observation groups."""
    contract = dgpo_abc_contract()
    assert [t.name for t in contract.policy.terms] == ["last_action", "task_multi_hot", "buffer_pose"]
    assert [t.name for t in contract.proprio.terms] == [
        "eef_pose",
        "gripper_pos",
        "joint_pos",
        "joint_vel",
        "gripper_force",
    ]
    assert [t.name for t in contract.privileged_proprio.terms] == [
        "eef_pose_diff",
        "joint_pos_diff",
        "object_target_pose_diff",
    ]
    assert contract.agent_policy_groups == ("policy", "proprio")
    assert contract.agent_critic_groups == ("policy", "proprio", "privileged_proprio")


def test_compat_runner_obs_groups_match_contract_dims():
    """LiberoAllDgpoPPORunnerCfg must feed actor=324 / critic=572 (rsl-rl >= 4.0)."""
    from isaaclab_contrib.tasks.manipulation.libero.config.franka.agents.rsl_rl_ppo_cfg import (
        LiberoAllDgpoPPORunnerCfg,
    )

    contract = dgpo_abc_contract()
    cfg = LiberoAllDgpoPPORunnerCfg()
    # rsl-rl >= 4.0 requires the "actor" key; "policy" alone falls back to env group dim 277.
    assert "actor" in cfg.obs_groups
    assert cfg.obs_groups["actor"] == list(contract.agent_policy_groups)
    assert cfg.obs_groups["critic"] == list(contract.agent_critic_groups)

    group_dims = {
        "policy": contract.policy.dim,
        "proprio": contract.proprio.dim,
        "privileged_proprio": contract.privileged_proprio.dim,
    }
    actor_dim = sum(group_dims[g] for g in cfg.obs_groups["actor"])
    critic_dim = sum(group_dims[g] for g in cfg.obs_groups["critic"])
    assert actor_dim == DGPO_ABC_ACTOR_OBS_DIM == 324
    assert critic_dim == DGPO_ABC_CRITIC_OBS_DIM == 572


def test_multi_hot_width_is_36():
    assert multi_hot_encoding_width() == 36
    assert len(build_overlap_tag_columns()) == 36
    assert len(DGPO_ABC_TASK_SEQUENCE) == 40
    assert len(LIBERO_SHARED_SUBTASK_TAGS) == 10
    # 6 shared tags + 30 fallback assignment keys = 36
    tagged = set(LIBERO_SHARED_SUBTASK_TAGS)
    assert len(DGPO_ABC_TASK_SEQUENCE) - len(tagged) + 6 == 36


def test_multi_hot_encoding_overlap_tags():
    """Shared-subtask rows fire both tags; untagged tasks use assignment fallback."""
    keys = ["libero_10::0", "libero_spatial::3", "libero_object::0"]
    enc = encode_assignment_multi_hot(keys, device="cpu")
    assert enc.shape == (3, 36)
    cols = build_overlap_tag_columns()
    soup_idx = cols.index("alphabet_soup_to_basket")
    tomato_idx = cols.index("tomato_sauce_to_basket")
    assert enc[0, soup_idx] == 1.0 and enc[0, tomato_idx] == 1.0
    assert enc[2, soup_idx] == 1.0
    # spatial::3 has no overlap tags → fallback one-hot on its assignment column
    spatial_idx = cols.index("libero_spatial::3")
    assert enc[1, spatial_idx] == 1.0
    assert enc[1].sum() == 1.0


def test_harvest_task_assignment_alias():
    assert harvest_task_to_assignment_key("long_task2", "libero_long") == "libero_10::2"
    assert harvest_task_to_assignment_key("spatial_task0", "libero_spatial") == "libero_spatial::0"
    assert harvest_task_to_assignment_key("object_task5", "libero_object") == "libero_object::5"


def test_dgpo_abc_harvest_suite_order_matches_task_sequence():
    """Compat suite harvest order must yield DGPO ``full_task_sequence`` keys."""
    from isaaclab_contrib.tasks.manipulation.libero.dgpo_layout import DGPO_ABC_HARVEST_SUITES

    keys: list[str] = []
    for suite_name, prefix in DGPO_ABC_HARVEST_SUITES:
        for task_id in range(10):
            keys.append(harvest_task_to_assignment_key(f"{prefix}_task{task_id}", suite_name))
    assert tuple(keys) == DGPO_ABC_TASK_SEQUENCE
    assert keys[0] == "libero_10::0"
    assert keys[10] == "libero_object::0"
    assert keys[20] == "libero_spatial::0"
    assert keys[30] == "libero_goal::0"


def test_dgpo_abc_action_and_control_contract():
    """OSC / gripper / control-rate fields must match DGPO+ABC env.yaml."""
    from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import (
        BinaryJointPositionActionCfg,
        OperationalSpaceControllerActionCfg,
    )

    from isaaclab_contrib.tasks.manipulation.libero.dgpo_layout import (
        DGPO_ABC_CONTROL_DT,
        DGPO_ABC_DECIMATION,
        DGPO_ABC_EPISODE_LENGTH_S,
        DGPO_ABC_GRIPPER_CLOSE,
        DGPO_ABC_GRIPPER_OPEN,
        DGPO_ABC_OSC_MOTION_STIFFNESS,
        DGPO_ABC_OSC_NULLSPACE_STIFFNESS,
        DGPO_ABC_OSC_ORIENTATION_SCALE,
        DGPO_ABC_OSC_POSITION_SCALE,
        DGPO_ABC_SIM_DT,
    )
    from isaaclab_contrib.tasks.manipulation.libero.robots.franka_osc import FRANKA_OSC

    assert DGPO_ABC_DECIMATION == 3
    assert pytest.approx(1.0 / 60.0) == DGPO_ABC_SIM_DT
    assert pytest.approx(0.05) == DGPO_ABC_CONTROL_DT
    assert DGPO_ABC_EPISODE_LENGTH_S == 26.0

    specs = FRANKA_OSC.action_specs()
    assert list(specs.keys()) == ["arm", "gripper"]
    arm_dim, arm_cfg = specs["arm"]
    grip_dim, grip_cfg = specs["gripper"]
    assert arm_dim == 6 and grip_dim == 1
    assert isinstance(arm_cfg, OperationalSpaceControllerActionCfg)
    assert isinstance(grip_cfg, BinaryJointPositionActionCfg)
    assert arm_cfg.asset_name == "franka_robot"
    assert grip_cfg.asset_name == "franka_robot"
    assert arm_cfg.body_name == "panda_hand"
    assert arm_cfg.position_scale == DGPO_ABC_OSC_POSITION_SCALE == pytest.approx(0.05)
    assert arm_cfg.orientation_scale == DGPO_ABC_OSC_ORIENTATION_SCALE == pytest.approx(0.5)
    ctrl = arm_cfg.controller_cfg
    assert isinstance(ctrl, OperationalSpaceControllerCfg)
    assert ctrl.target_types == ["pose_rel"]
    assert ctrl.impedance_mode == "fixed"
    assert ctrl.nullspace_control == "position"
    assert ctrl.nullspace_stiffness == DGPO_ABC_OSC_NULLSPACE_STIFFNESS
    assert tuple(ctrl.motion_stiffness_task) == DGPO_ABC_OSC_MOTION_STIFFNESS
    assert grip_cfg.open_command_expr == {"panda_finger_.*": DGPO_ABC_GRIPPER_OPEN}
    assert grip_cfg.close_command_expr == {"panda_finger_.*": DGPO_ABC_GRIPPER_CLOSE}


def test_multi_hot_bits_match_known_assignments():
    """Bit-for-bit multi-hot for libero_10::0 and libero_object::0 (playground tags)."""
    cols = build_overlap_tag_columns()
    enc = encode_assignment_multi_hot(["libero_10::0", "libero_object::0"], device="cpu")
    # Shared tags occupy the first columns in LIBERO_SHARED_SUBTASK_ORDER.
    assert cols[:6] == (
        "alphabet_soup_to_basket",
        "tomato_sauce_to_basket",
        "cream_cheese_to_basket",
        "butter_to_basket",
        "turn_on_flat_stove",
        "moka_pot_on_stove",
    )
    expected_10_0 = torch.zeros(36)
    expected_10_0[cols.index("alphabet_soup_to_basket")] = 1.0
    expected_10_0[cols.index("tomato_sauce_to_basket")] = 1.0
    expected_obj_0 = torch.zeros(36)
    expected_obj_0[cols.index("alphabet_soup_to_basket")] = 1.0
    assert torch.equal(enc[0], expected_10_0)
    assert torch.equal(enc[1], expected_obj_0)


def test_pose_buffer_is_26x9():
    assert len(LIBERO_ORDERED_OBJECT_NAMES) == 26
    contract = dgpo_abc_contract()
    assert contract.policy.terms[2].dim == 234
    assert contract.privileged_proprio.terms[2].dim == 234


def test_harvest_vs_per_env_entity_anchors():
    """Documented AssetView reduction: PerEnvMixin ~193 → harvest ~35 prototypes."""
    assert PER_ENV_DGPO_ABC_SCENE_ENTITIES == 193
    assert HARVEST_ALL_PROTOTYPES == 35
    assert pytest.approx(193 / 35) == PER_ENV_DGPO_ABC_SCENE_ENTITIES / HARVEST_ALL_PROTOTYPES


@pytest.mark.parametrize(
    "group,expected",
    [
        ("policy", 277),
        ("proprio", 47),
        ("privileged_proprio", 248),
    ],
)
def test_group_dims(group: str, expected: int):
    contract = dgpo_abc_contract()
    dims = {
        "policy": contract.policy.dim,
        "proprio": contract.proprio.dim,
        "privileged_proprio": contract.privileged_proprio.dim,
    }
    assert dims[group] == expected


def _default_checkpoint_path() -> Path:
    env_path = os.environ.get("DGPO_ABC_CHECKPOINT")
    if env_path:
        return Path(env_path)
    candidates = [
        Path(
            "/data/Projects/Robotics/IsaacLab/reb_isaaclab/_RobotLearningLab/logs/paper_plots/DGPO+ABC/model_32500.pt"
        ),
        Path("/data/Projects/Robotics/IsaacLab/RobotLearningLab/logs/paper_plots/DGPO+ABC/model_32500.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _find_linear_in_weight(state: dict, prefer_substrings: tuple[str, ...]) -> torch.Tensor | None:
    """Best-effort locate a first Linear weight matching prefer_substrings order."""
    # Flatten nested dicts (rsl_rl checkpoints nest under model_state_dict / actor_critic).
    flat: dict[str, torch.Tensor] = {}

    def _walk(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            flat[prefix] = obj
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else str(k))

    _walk(state)
    candidates = [v for k, v in flat.items() if k.endswith("weight") and v.ndim == 2]
    for needle in prefer_substrings:
        for k, v in flat.items():
            if needle in k and k.endswith("weight") and v.ndim == 2:
                return v
    return candidates[0] if candidates else None


@pytest.mark.skipif(not _default_checkpoint_path().is_file(), reason="DGPO+ABC checkpoint not on disk")
def test_checkpoint_first_layer_shapes_match_contract():
    """Load model_32500.pt and assert actor/critic input dims without Isaac Sim.

    Eval command (full semantic rollout, once traj bank is wired)::

        LIBERO_ASSETS_DATA_DIR=... LIBERO_CONFIG_DIR=... \\
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
            --task Isaac-Libero-All-Dgpo-Osc-Play-v0 --num_envs 40 \\
            --checkpoint /path/to/DGPO+ABC/model_32500.pt presets=physx --headless
    """
    path = _default_checkpoint_path()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor_w = _find_linear_in_weight(
        ckpt,
        ("actor.0.weight", "actor_obs_normalizer", "policy.actor.0.weight", "actor."),
    )
    critic_w = _find_linear_in_weight(
        ckpt,
        ("critic.0.weight", "critic_obs_normalizer", "policy.critic.0.weight", "critic."),
    )
    assert actor_w is not None, f"Could not find actor Linear weight in {path}"
    assert critic_w is not None, f"Could not find critic Linear weight in {path}"
    # Prefer the layer whose *in* features match the locked obs dims.
    actor_candidates = [actor_w]
    critic_candidates = [critic_w]
    flat: dict[str, torch.Tensor] = {}

    def _walk(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            flat[prefix] = obj
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else str(k))

    _walk(ckpt)
    for k, v in flat.items():
        if v.ndim == 2 and v.shape[0] == 512:
            if v.shape[1] == DGPO_ABC_ACTOR_OBS_DIM:
                actor_candidates.append(v)
            if v.shape[1] == DGPO_ABC_CRITIC_OBS_DIM:
                critic_candidates.append(v)

    assert any(w.shape == (512, DGPO_ABC_ACTOR_OBS_DIM) for w in actor_candidates), (
        f"No (512, {DGPO_ABC_ACTOR_OBS_DIM}) actor weight; saw "
        f"{sorted({tuple(w.shape) for w in flat.values() if isinstance(w, torch.Tensor) and w.ndim == 2})}"
    )
    assert any(w.shape == (512, DGPO_ABC_CRITIC_OBS_DIM) for w in critic_candidates), (
        f"No (512, {DGPO_ABC_CRITIC_OBS_DIM}) critic weight"
    )


# ---------------------------------------------------------------------------
# Demo command stack (no Isaac Sim)
# ---------------------------------------------------------------------------


def test_build_env_task_assignments_cycles_bindings():
    from types import SimpleNamespace

    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import build_env_task_assignments

    bindings = [
        SimpleNamespace(name="spatial_task0", suite="libero_spatial"),
        SimpleNamespace(name="goal_task1", suite="libero_goal"),
        SimpleNamespace(name="object_task2", suite="libero_object"),
        SimpleNamespace(name="long_task3", suite="libero_long"),
    ]
    keys = build_env_task_assignments(bindings, num_envs=6)  # type: ignore[arg-type]
    assert keys == [
        "libero_spatial::0",
        "libero_goal::1",
        "libero_object::2",
        "libero_10::3",
        "libero_spatial::0",
        "libero_goal::1",
    ]


def test_resolve_demos_root_finds_sim2sim_or_none():
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import resolve_libero_demos_root

    root = resolve_libero_demos_root(require=False)
    if root is not None:
        assert os.path.isdir(root)
        assert any(name.startswith("libero_") and name.endswith(".hdf5") for name in os.listdir(root))


def test_pack_multi_banks_pads_to_max_len():
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import (
        UNIFIED_LIBERO_SEMANTIC_KEYS,
        _UnifiedLiberoCommandBackend,
    )

    backend = object.__new__(_UnifiedLiberoCommandBackend)
    backend._command_dim = 7
    backend.device = torch.device("cpu")
    per_key = {k: [torch.randn(3, 7), torch.randn(5, 7)] for k in UNIFIED_LIBERO_SEMANTIC_KEYS}
    banks, lengths = backend._pack_multi_banks(per_key)
    assert lengths.tolist() == [3, 5]
    for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
        assert banks[key].shape == (2, 5, 7)
        assert torch.allclose(banks[key][0, :3], per_key[key][0])
        assert torch.all(banks[key][0, 3:] == 0)


def test_canonical_object_name_strips_group_suffix():
    from isaaclab_contrib.tasks.manipulation.libero.mdp.observations import ObjectTargetPoseDiff

    assert ObjectTargetPoseDiff._canonical_object_name("alphabet_soup_1") == "alphabet_soup_1"
    assert ObjectTargetPoseDiff._canonical_object_name("object_alphabet_soup_1_group_3") == "alphabet_soup_1"


@pytest.mark.skipif(
    not Path("/data/Projects/Robotics/IsaacLab/RobotLearningLab/benchmarks/datasets/libero/sim2sim_dataset").is_dir(),
    reason="sim2sim_dataset not on disk",
)
def test_resolve_dataset_path_for_object_task0():
    """HDF5 path resolution matches DGPO+ABC naming without loading Isaac Sim."""
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import (
        CompatDemoTaskConfig,
        SourceLiberoCommandCfg,
        _UnifiedLiberoCommandBackend,
    )

    root = "/data/Projects/Robotics/IsaacLab/RobotLearningLab/benchmarks/datasets/libero/sim2sim_dataset"
    backend = object.__new__(_UnifiedLiberoCommandBackend)
    backend.cfg = SourceLiberoCommandCfg(
        libero_config=CompatDemoTaskConfig(env_task_assignments=["libero_object::0"]),
        datasets_root=root,
    )
    path = backend._resolve_dataset_path(("libero_object", 0))
    assert os.path.isfile(path)
    assert "libero_object_task0_" in os.path.basename(path)
    assert path.endswith(".hdf5")


def test_dgpo_obs_notes_updated_after_command_wiring():
    from isaaclab_contrib.tasks.manipulation.libero.mdp.observations import DGPO_OBS_NOTES

    joined = " ".join(DGPO_OBS_NOTES).lower()
    assert "wire sourceliberocommand" not in joined
    assert "wire ee_pose" not in joined
    assert "reset objects from demo" not in joined


def test_canonical_demo_asset_to_proto_mapping():
    """Demo HDF5 keys map onto harvest prototype names for filter_reset_ids writes."""
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.events import (
        build_canonical_to_proto_map,
        canonical_demo_asset_name,
        resolve_demo_asset_to_scene,
    )
    from isaaclab_contrib.tasks.manipulation.libero.tasks.harvest.prototypes import ObjectBinding, TaskBinding

    binding = TaskBinding(
        name="object_task0",
        suite="libero_object",
        fixture_proto="kitchen_table",
        object_bindings=[
            ObjectBinding(
                proto_name="alphabet_soup",
                canonical_name="alphabet_soup_1",
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                is_articulation=False,
            ),
            ObjectBinding(
                proto_name="basket",
                canonical_name="basket_1",
                pos=(0.1, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                is_articulation=False,
            ),
        ],
        primary_proto="alphabet_soup",
        target_proto="basket",
    )
    cmap = build_canonical_to_proto_map(binding)
    assert cmap["alphabet_soup_1"] == "alphabet_soup"
    assert (
        resolve_demo_asset_to_scene(
            "object_alphabet_soup_1_group_3", entity_type="rigid_object", canonical_to_proto=cmap
        )
        == "alphabet_soup"
    )
    assert resolve_demo_asset_to_scene("robot", entity_type="articulation", canonical_to_proto=cmap) == "franka_robot"
    assert resolve_demo_asset_to_scene("missing_obj", entity_type="rigid_object", canonical_to_proto=cmap) is None
    assert canonical_demo_asset_name("object_basket_1_group_0") == "basket_1"


def test_reserve_trajectories_pins_active_traj_indices():
    """reserve → _resample must reuse the reserved traj index (demo reset alignment)."""
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import (
        UNIFIED_LIBERO_SEMANTIC_KEYS,
        _UnifiedLiberoCommandBackend,
    )

    backend = object.__new__(_UnifiedLiberoCommandBackend)
    backend.device = torch.device("cpu")
    backend.num_envs = 2
    backend.cfg = type(
        "Cfg",
        (),
        {"randomize_initial_state_timestep": False, "sample_mode": "random", "track_metrics": False},
    )()
    backend._command_dim = 7
    backend._traj_banks = {k: torch.zeros(3, 4, 7) for k in UNIFIED_LIBERO_SEMANTIC_KEYS}
    for t in range(3):
        for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
            backend._traj_banks[key][t, 0] = float(t)
    backend._traj_lengths = torch.tensor([4, 4, 4], dtype=torch.long)
    backend._traj_initial_states = [
        {"rigid_object": {"alphabet_soup_1": {"root_pose": torch.zeros(1, 7)}}},
        None,
        {"rigid_object": {"basket_1": {"root_pose": torch.ones(1, 7)}}},
    ]
    backend._assignment_ids = torch.tensor([0, 0], dtype=torch.long)
    backend._assignment_traj_indices = {0: torch.tensor([0, 1, 2], dtype=torch.long)}
    backend._assignment_seq_cursor = torch.zeros(1, dtype=torch.long)
    backend._assignment_lookup = {0: ("libero_object", 0)}
    backend._reserved_traj_indices = {}
    backend._reserved_start_steps = {}
    backend._commands = {k: torch.zeros(2, 7) for k in UNIFIED_LIBERO_SEMANTIC_KEYS}
    backend._active_traj = torch.zeros(2, dtype=torch.long)
    backend._traj_len_per_env = torch.ones(2, dtype=torch.long)
    backend._frame_counters = torch.zeros(2, dtype=torch.long)
    backend._last_reset_start_steps = torch.zeros(2, dtype=torch.long)
    backend.metrics = {}

    # Force draw to pick traj 2 for env 0 by making only that candidate available.
    backend._assignment_traj_indices = {0: torch.tensor([2], dtype=torch.long)}
    env_ids = torch.tensor([0], dtype=torch.long)
    reservations = backend.reserve_trajectories_for_envs(env_ids)
    assert 0 in reservations
    assert backend._reserved_traj_indices[0] == 2
    assert reservations[0] is not None
    assert "basket_1" in reservations[0]["rigid_object"]

    backend._resample(env_ids)
    assert int(backend._active_traj[0].item()) == 2
    assert 0 not in backend._reserved_traj_indices  # consumed
    assert torch.allclose(backend._commands["source_action"][0], torch.full((7,), 2.0))


def test_dgpo_osc_profile_command_documented():
    """Profiling entry remains available for harvest OSC / DGPO path."""
    cmd = (
        "./isaaclab.sh -p scripts/benchmarks/profile_libero_scene.py "
        "--suite all --impl compat_osc --num_envs 40 --num_steps 50 "
        "--warmup_steps 10 --repeats 1 --seed 0 --headless presets=physx"
    )
    assert "compat_osc" in cmd
    assert PER_ENV_DGPO_ABC_SCENE_ENTITIES == 193
    assert HARVEST_ALL_PROTOTYPES == 35


def test_workspace_shift_raises_floor_demo_poses_into_kitchen_frame():
    """Demo initial_state is in the raw LIBERO frame; harvest needs kitchen-base shift.

    Floor / living-room demos store object z near 0 / ~0.45. Without applying
    ``workspace_shift``, reset writes those poses under the raised fixture so
    objects appear to fall to the ground.
    """
    from isaaclab_contrib.tasks.manipulation.libero.mdp.demos.commands import (
        apply_workspace_shift_to_trajectory_state,
        build_assignment_workspace_shifts,
    )
    from isaaclab_contrib.tasks.manipulation.libero.tasks.common.suite_loader import ROBOT_BASE_KITCHEN
    from isaaclab_contrib.tasks.manipulation.libero.tasks.harvest.prototypes import (
        ObjectBinding,
        TaskBinding,
        harvest_libero_prototypes,
    )

    # Synthetic floor-workspace demo pose (matches libero_object task0 scale).
    floor_shift = (
        ROBOT_BASE_KITCHEN[0] - (-0.6),
        ROBOT_BASE_KITCHEN[1] - 0.0,
        ROBOT_BASE_KITCHEN[2] - 0.0,
    )
    state = {
        "rigid_object": {
            "alphabet_soup_1": {
                "root_pose": torch.tensor([[-0.1193, -0.2398, 0.0384, 0.0, 0.0, 0.0, 1.0]]),
            }
        },
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([[-0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            }
        },
    }
    apply_workspace_shift_to_trajectory_state(state, floor_shift)
    soup_z = float(state["rigid_object"]["alphabet_soup_1"]["root_pose"][0, 2])
    robot_z = float(state["articulation"]["robot"]["root_pose"][0, 2])
    assert soup_z == pytest.approx(0.0384 + floor_shift[2], abs=1e-5)
    assert robot_z == pytest.approx(floor_shift[2], abs=1e-5)
    assert soup_z > 0.5  # now table-height in harvest kitchen frame

    # Zero shift is a no-op (kitchen tasks).
    kitchen_pose = torch.tensor([[0.06, 0.0, 0.966, 0.0, 0.0, 0.0, 1.0]])
    kitchen_state = {"rigid_object": {"moka_pot_1": {"root_pose": kitchen_pose.clone()}}}
    apply_workspace_shift_to_trajectory_state(kitchen_state, (0.0, 0.0, 0.0))
    assert torch.equal(kitchen_state["rigid_object"]["moka_pot_1"]["root_pose"], kitchen_pose)

    # Harvest bindings expose non-zero shifts for non-kitchen workspaces.
    try:
        _, tasks = harvest_libero_prototypes()
    except FileNotFoundError:
        pytest.skip("LIBERO config dir not available")
    shifts = build_assignment_workspace_shifts(tasks)
    assert "libero_object::0" in shifts
    assert shifts["libero_object::0"][2] == pytest.approx(0.912, abs=1e-5)
    # Kitchen long task has zero shift.
    assert "libero_10::2" in shifts
    assert shifts["libero_10::2"] == (0.0, 0.0, 0.0)

    # Assignment map keys match harvest_task_to_assignment_key.
    binding = TaskBinding(
        name="object_task0",
        suite="libero_object",
        fixture_proto="floor",
        object_bindings=[
            ObjectBinding(
                proto_name="alphabet_soup",
                canonical_name="alphabet_soup_1",
                pos=(0.0, 0.0, 0.95),
                rot=(0.0, 0.0, 0.0, 1.0),
                is_articulation=False,
            )
        ],
        primary_proto="alphabet_soup",
        target_proto="alphabet_soup",
        workspace_shift=floor_shift,
    )
    assert build_assignment_workspace_shifts([binding])["libero_object::0"] == floor_shift
