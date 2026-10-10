# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shoelace runtime configuration and randomized resets."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.shoelace import shoelace_constants as constants
from isaaclab_tasks.contrib.shoelace.mdp.events import reset_shoe_position
from isaaclab_tasks.contrib.shoelace.mdp.grasp import shoelace_grasp_quality
from isaaclab_tasks.contrib.shoelace.mdp.observations import gripper_close_error, tail_tcp_relative_speed, tails_to_tcp
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import configure_shoelace_builder, create_shoelace_env


@pytest.mark.parametrize(("open_position", "closed_position", "direction"), [(0.04, 0.0, -1.0), (0.0, 1.2, 1.0)])
def test_finger_closure_quality_and_residual_support_both_joint_directions(
    open_position: float, closed_position: float, direction: float
) -> None:
    """Closing prismatic and revolute fingers must gain grasp quality and reduce obstruction residuals."""
    positions = torch.tensor([[open_position], [(open_position + closed_position) / 2], [closed_position]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=positions),
            joint_pos_target=SimpleNamespace(torch=torch.full_like(positions, closed_position)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(3, 1, 4)),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(3, 1, 6)),
        )
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=torch.zeros(3, 3, 7)),
            segment_velocity_w=SimpleNamespace(torch=torch.zeros(3, 3, 6)),
        )
    )
    env = SimpleNamespace(
        num_envs=3,
        cfg=SimpleNamespace(),
        scene={
            "robot": robot,
            "left": cable,
            "right": cable,
            "finger_tail_contacts": SimpleNamespace(data=torch.zeros(3, 4)),
        },
    )
    robot_cfgs = (SceneEntityCfg("robot", joint_ids=[0], body_ids=[0]),) * 2
    quality, finite = shoelace_grasp_quality(
        env, 0.001, 0.08, open_position, closed_position, (SceneEntityCfg("left"), SceneEntityCfg("right")), robot_cfgs
    )
    torch.testing.assert_close(quality, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]), atol=1.0e-5, rtol=0.0)
    assert finite.all()
    residual = gripper_close_error(env, robot_cfgs, closing_direction=direction) / abs(closed_position - open_position)
    torch.testing.assert_close(residual, torch.tensor([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]))


def test_tcp_observations_use_each_hand_offset_for_position_and_angular_velocity() -> None:
    """Distinct hand TCPs must rotate their positions and include angular motion when measuring slip."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            body_pos_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 2**-0.5, 2**-0.5]]])),
            body_link_vel_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]])),
        )
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=torch.zeros(1, 3, 7)),
            segment_velocity_w=SimpleNamespace(torch=torch.zeros(1, 3, 6)),
        )
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(tcp_offsets=((0.1, 0.0, 0.0), (0.0, 0.2, 0.0))),
        scene={"robot": robot, "left": cable, "right": cable},
    )
    robot_cfgs = (SceneEntityCfg("robot", body_ids=[0]),) * 2
    cable_cfgs = (SceneEntityCfg("left"), SceneEntityCfg("right"))
    torch.testing.assert_close(
        tails_to_tcp(env, cable_cfgs, robot_cfgs), torch.tensor([[0.0, -0.1, 0.0, 0.2, 0.0, 0.0]])
    )
    torch.testing.assert_close(tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs), torch.tensor([[0.2, 0.4]]))


@pytest.mark.parametrize("env_ids", [slice(None), slice(1, 4, 2), slice(0, 0), torch.tensor([3, 1])])
def test_shoe_reset_accepts_slices_and_preserves_unselected_poses(env_ids: torch.Tensor | slice) -> None:
    """Apply the same translation to selected shoe and cable poses without expanding slices."""
    defaults = torch.zeros(4, 7)
    defaults[:, 6] = 1.0
    origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shoe_pose = defaults.clone()
    cable_poses = [defaults[:, None].expand(-1, 3, -1).clone() for _ in range(2)]
    cable_poses[1][..., 1] = 0.5
    initial_cable_poses = [pose.clone() for pose in cable_poses]
    shoe = SimpleNamespace(
        data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=defaults)),
        write_root_pose_to_sim_index=lambda *, root_pose, env_ids: shoe_pose.__setitem__(env_ids, root_pose),
    )
    cables = [
        SimpleNamespace(
            data=SimpleNamespace(default_segment_pose_w=SimpleNamespace(torch=pose.clone())),
            write_segment_pose_to_sim_index=lambda *, segment_pose, env_ids, buffer=pose: buffer.__setitem__(
                env_ids, segment_pose
            ),
        )
        for pose in cable_poses
    ]

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(device="cpu", scene=Scene(shoe=shoe, shoelace_left=cables[0], shoelace_right=cables[1]))
    reset_shoe_position(env, env_ids, {"x": (0.1, 0.1), "y": (-0.2, -0.2)})
    expected_shoe = defaults.clone()
    offset = torch.tensor([0.1, -0.2, 0.0])
    expected_shoe[env_ids, :3] += origins[env_ids] + offset
    torch.testing.assert_close(shoe_pose, expected_shoe)
    for actual, expected in zip(cable_poses, initial_cable_poses, strict=True):
        expected[env_ids, :, :3] += offset
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("num_envs", "override"), [(4, None), (1024, None), (1024, 1_024_000)])
def test_runtime_contact_capacity_scales_and_preserves_overrides(num_envs: int, override: int | None) -> None:
    """Grow collision storage while keeping matching indices encodable and honoring larger overrides."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = num_envs
    outer_override = 16_777_216 if override is not None else 0
    if override is not None:
        cfg.sim.physics.collision_cfg.max_triangle_pairs = outer_override
        cfg.sim.physics.solver_cfg.contact_max_triangle_pairs = override
        cfg.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor = 4.0
    cfg.validate()

    collision = cfg.sim.physics.collision_cfg
    solver = cfg.sim.physics.solver_cfg
    assert collision.max_triangle_pairs >= constants.TRIANGLE_PAIRS_PER_ENV * num_envs
    assert 0 < solver.contact_max_triangle_pairs < 2**20
    if override is not None:
        assert collision.max_triangle_pairs == outer_override
        assert solver.contact_max_triangle_pairs == override
        assert solver.contact_reduction_hashtable_size_factor == 4.0
    elif num_envs == 4:
        assert solver.contact_max_triangle_pairs == collision.max_triangle_pairs
    else:
        assert solver.contact_max_triangle_pairs < collision.max_triangle_pairs
        assert solver.contact_reduction_hashtable_size_factor > 0.25


@pytest.mark.parametrize("regularization", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_proxy_inertia_is_rejected_before_environment_construction(
    regularization: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid builder regularization at the factory entry before initializing an environment."""
    cfg = ShoelaceEnvCfg()
    cfg.cable_inertia_regularization = regularization
    monkeypatch.setattr(
        "isaaclab_tasks.contrib.shoelace.shoelace_physics.ManagerBasedRLEnv",
        lambda **kwargs: pytest.fail("Invalid regularization must not initialize an environment"),
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        create_shoelace_env(cfg=cfg)


def test_cable_friction_uses_each_bound_usd_material() -> None:
    """Preserve distinct authored cable friction through model construction without external assets."""
    stage = Usd.Stage.CreateInMemory()
    builder = newton.ModelBuilder()
    builder.begin_world()
    expected = {}
    for side, friction in (("Left", 0.23), ("Right", 0.37)):
        path = f"/World/envs/env_0/ShoelaceScene/Shoelace{side}/geometry/mesh"
        curve = UsdGeom.BasisCurves.Define(stage, path).GetPrim()
        points = np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0)])
        segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()
        for name in ("referenceSegmentLength", "segmentLength"):
            curve.CreateAttribute(f"shoelace:{name}", Sdf.ValueTypeNames.Double).Set(segment_length)
        material = UsdShade.Material.Define(stage, f"{path}/ContactMaterial")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(friction)
        UsdShade.MaterialBindingAPI.Apply(curve).Bind(material, materialPurpose="physics")
        bodies, _ = builder.add_rod(rod=newton.Rod(points, radius=0.001), label=path, body_frame_origin="start")
        for body in bodies:
            expected.update({shape: friction for shape in builder.body_shapes[body]})
    builder.end_world()

    configure_shoelace_builder(builder, stage, 0, np.array([0.0, 0.0, 0.0, 1.0]), ShoelaceEnvCfg())
    model = builder.finalize(device="cpu")
    for shape, friction in expected.items():
        assert model.shape_material_mu.numpy()[shape] == pytest.approx(friction)
