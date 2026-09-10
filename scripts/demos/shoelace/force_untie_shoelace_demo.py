# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tighten and untie an authored shoelace with force-controlled Newton VBD dynamics.

Only the virtual fingertip forces are scripted. The bow loops and tails stay dynamic, and no pose
or velocity target is tracked. Two fixed endpoint segments anchor the eyelet-held span, whose
interior uses one visible static collision mesh.

``--cable_inertia_regularization`` adds isotropic proxy inertia [kg*m^2] to dynamic
cable segments only (default: 1e-6), including randomized-radius segments. It leaves
masses and anchors unchanged. Zero disables this addition, not Newton's own validation.

.. code-block:: bash

    # Run the complete 12 second sequence with the Newton visualizer.
    uv run python scripts/demos/shoelace/force_untie_shoelace_demo.py

    # Run 1024 isolated Newton worlds in parallel (a large-memory workload).
    uv run python scripts/demos/shoelace/force_untie_shoelace_demo.py --num_envs 1024 --visualizer none

    # Randomize shoe color, cable color, and cable radius once per environment.
    uv run python scripts/demos/shoelace/force_untie_shoelace_demo.py --num_envs 16 --randomize --randomization_seed 7

    # Keep the authored shoelace pose by disabling gravity.
    uv run python scripts/demos/shoelace/force_untie_shoelace_demo.py --disable_gravity

"""

from __future__ import annotations

import argparse
import colorsys
import math
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-controlled Newton VBD shoelace demo.", conflict_handler="resolve")
parser.add_argument("--max_steps", type=int, default=720, help="Number of 60 Hz simulation steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of isolated Newton worlds.")
parser.add_argument(
    "--cable_inertia_regularization",
    type=float,
    default=1.0e-6,
    help="Isotropic proxy inertia added per dynamic cable segment [kg*m^2].",
)
parser.add_argument(
    "--randomize",
    action="store_true",
    help="Randomize shoe color, cable color, and cable radius once per environment at startup.",
)
parser.add_argument("--randomization_seed", type=int, default=0, help="Seed for per-environment randomization.")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity for this demo.")
add_launcher_args(parser)
parser.set_defaults(physics="newton_vbd", visualizer=["newton_gl"])
args_cli = parser.parse_args()

import numpy as np
import torch
import warp as wp
from _newton_force_schedule import BodyForceMove, OpenLoopBodyForceSchedule
from _newton_shoelace import (
    BEND_DAMPING,
    BEND_STIFFNESS,
    CABLE_INERTIA_REGULARIZATION,
    CONTACT_GAP,
    CONTACT_KD,
    CONTACT_KE,
    CONTACTS_PER_ENV,
    LACE_MU,
    MIN_TRIANGLE_PAIRS,
    PINNED_FIRST,
    PINNED_LAST,
    STRETCH_DAMPING,
    STRETCH_STIFFNESS,
    TRIANGLE_PAIRS_PER_ENV,
    VBD_CONTACT_BUFFER,
    cable_joint_dof_slice,
    cable_spawn_cfg,
    configure_dahl,
    configure_shoelace_builder,
    ground_asset_cfg,
    ground_size,
    install_shoe_material_reference,
    light_asset_cfg,
    load_shoelace,
    parallel_transport_normals,
    pinned_shoelace_spawner,
    shoe_asset_cfg,
    shoe_collider_asset_cfg,
    shoe_visual_asset_cfg,
    tongue_upper_asset_cfg,
)
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, VisualMaterial, VisualMaterialCfg
from isaaclab.physics import PhysicsCfg, PhysicsEvent
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from newton import State

# Simulation cadence.
FPS = 60
SIM_SUBSTEPS = 10
SIM_ITERATIONS = 20  # VBD solve iterations per substep after contacts are generated.
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS
ENV_SPACING = 0.5

# Lace stiffnesses and damping below are Newton per-joint values.
CABLE_JOINT_STIFFNESS = (STRETCH_STIFFNESS, STRETCH_STIFFNESS, BEND_STIFFNESS, BEND_STIFFNESS)
CABLE_JOINT_DAMPING = (STRETCH_DAMPING, STRETCH_DAMPING, BEND_DAMPING, BEND_DAMPING)

# Simulation-only contact configuration.
GRAVITY = -9.81

# Tightening and untying force schedule [s].
SETTLE_END = 0.6
TIGHTEN_END = 1.6
RELEASE_TIME = 2.1
TIGHTEN_LIFT_END = 1.1
CABLE_RADIUS_BUCKETS = 5
CABLE_RADIUS_SCALE_RANGE = (0.85, 1.15)
LOOP_GRAB_LEFT = (42, 46)
LOOP_GRAB_RIGHT = (314, 318)
UNTIE_FIRST_START = 3.0
UNTIE_FIRST_END = 4.2
UNTIE_SECOND_START = 5.2
UNTIE_SECOND_END = 6.4
UNTIE_HOLD = 0.3
UNTIE_RELAX = 0.6
# The imported Lab contact trajectory can leave the loosened bight at the throat. A final force-only
# extraction gives the standalone demo a reproducible untied terminal state without adding a pose target.
EXTRACT_START = 7.4
EXTRACT_END = 9.6
EXTRACT_FORCE = 0.1
EXTRACT_DOWNWARD_BIAS = 1.0
TAIL_GRAB_FIRST = (357, 360)
TAIL_GRAB_SECOND = (0, 3)
TIGHTEN_FORCE_RAMP = 1.0
TIGHTEN_LIFT_RAMP = 0.2
UNTIE_FORCE_RAMP = 1.0
TIGHTEN_FORCE = 0.1
TIGHTEN_LIFT_FORCE = 0.01
UNTIE_FORCE_FIRST = 0.1
UNTIE_FORCE_SECOND = 0.1

CAMERA_EYE = (0.2159324, -0.2158787, 0.2631368)
CAMERA_TARGET = (-0.0000817, 0.0001354, 0.0722455)


def _sample_randomization(num_envs: int, cable_radius: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-environment linear RGB colors and bucketed cable radii [m]."""
    rng = np.random.default_rng(seed)
    shoe_colors = np.asarray(
        [colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.45, 0.80), rng.uniform(0.45, 0.85)) for _ in range(num_envs)],
        dtype=np.float32,
    )
    radius_scales = np.linspace(*CABLE_RADIUS_SCALE_RANGE, CABLE_RADIUS_BUCKETS)
    cable_radii = cable_radius * rng.choice(radius_scales, size=num_envs)
    cable_colors = np.asarray(
        [colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.35, 0.75), rng.uniform(0.25, 0.70)) for _ in range(num_envs)],
        dtype=np.float32,
    )
    return shoe_colors, cable_colors, cable_radii


class ShoelaceController:
    """Configure replicated Newton shoelaces and apply virtual fingertip forces.

    Args:
        centerline: Resampled cable centerline [m], shape [N, 3].
        cable_radius: Authored cable radius [m].
        num_envs: Number of isolated Newton worlds.
        cable_radii: Optional randomized cable radii [m], shape [num_envs].
        cable_colors: Optional randomized linear RGB colors, shape [num_envs, 3].
        cable_inertia_regularization: Isotropic proxy inertia added per dynamic segment [kg*m^2].
    """

    def __init__(
        self,
        centerline: np.ndarray,
        cable_radius: float,
        num_envs: int,
        cable_radii: np.ndarray | None = None,
        cable_colors: np.ndarray | None = None,
        cable_inertia_regularization: float = CABLE_INERTIA_REGULARIZATION,
    ) -> None:
        """Initialize authored geometry and optional per-environment variations."""
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.cable_inertia_regularization = cable_inertia_regularization
        self.cable_radii = None if cable_radii is None else np.asarray(cable_radii, dtype=np.float64)
        if self.cable_radii is not None and self.cable_radii.shape != (num_envs,):
            raise ValueError(f"Expected {num_envs} cable radii, got shape {self.cable_radii.shape}")
        self.cable_colors = None if cable_colors is None else np.asarray(cable_colors, dtype=np.float32)
        if self.cable_colors is not None and self.cable_colors.shape != (num_envs, 3):
            raise ValueError(f"Expected cable colors with shape {(num_envs, 3)}, got {self.cable_colors.shape}")
        self.cable_bodies: list[list[int]] = []
        self.cable_joints: list[list[int]] = []
        self._initial_body_q = None
        self._force_schedule: OpenLoopBodyForceSchedule | None = None

    def register(self) -> None:
        """Register model lifecycle and per-substep force callbacks."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_builder")
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY, name="shoelace_model")
        NewtonManager.register_state_force_callback(self._apply_forces)

    def _configure_builder(self, _: object) -> None:
        """Finalize replicated cable geometry, materials, topology, and collision filters."""
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton builder is unavailable during MODEL_INIT")
        build = configure_shoelace_builder(
            builder,
            self.centerline,
            self.cable_radius,
            self.num_envs,
            self.cable_radii,
            self.cable_colors,
            write_shape_colors=True,
            cable_inertia_regularization=self.cable_inertia_regularization,
        )
        self.cable_bodies = [left + right for left, right in build.body_chains]
        self.cable_joints = [left + right for left, right in build.joint_chains]
        for world_joints in build.joint_chains:
            for joints in world_joints:
                dof_slice = cable_joint_dof_slice(builder, joints)
                builder.joint_target_ke[dof_slice] = CABLE_JOINT_STIFFNESS * len(joints)
                builder.joint_target_kd[dof_slice] = CABLE_JOINT_DAMPING * len(joints)
        configure_dahl(builder, self.cable_joints)

    def _configure_model(self, _: object) -> None:
        """Capture the initial state and prepare the force schedule."""
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        self._initial_body_q = wp.clone(state.body_q)
        moves = self._build_force_moves()
        self._force_schedule = OpenLoopBodyForceSchedule(
            moves,
            device=str(model.device),
            sim_dt=SIM_DT,
        )

    def restore_authored_state(self) -> None:
        """Restore the authored knot and clear velocities, forces, contacts, and schedule state."""
        if self._initial_body_q is None or self._force_schedule is None:
            raise RuntimeError("The shoelace model was not initialized")

        model = NewtonManager.get_model()
        rest = model.body_q.numpy()
        for bodies in self.cable_bodies:
            for chain in (bodies[: PINNED_FIRST + 1], bodies[PINNED_FIRST + 1 :]):
                rest[np.asarray(chain), 3:7] = rest[chain[0], 3:7]
        model.body_q.assign(rest)
        state_0 = NewtonManager.get_state_0()
        state_1 = NewtonManager.get_state_1()
        states = (state_0,) if state_1 is None or state_1 is state_0 else (state_0, state_1)
        for state in states:
            state.body_q.assign(self._initial_body_q)
            state.body_qd.zero_()
            state.clear_forces()
        self._force_schedule.reset()
        NewtonManager.invalidate_body_state()

    def _apply_forces(self, state: State) -> None:
        """Apply scheduled forces before one Newton solver substep."""
        if self._force_schedule is None:
            raise RuntimeError("The shoelace force schedule was not initialized")
        self._force_schedule.apply(state)

    def _build_force_moves(self) -> list[BodyForceMove]:
        """Build the complete tightening and untying force sequence for every world."""
        midpoints = 0.5 * (self.centerline[:-1] + self.centerline[1:])
        knot = 0.5 * (midpoints[PINNED_FIRST] + midpoints[PINNED_LAST])
        moves: list[BodyForceMove] = []
        for world in range(self.num_envs):
            for lower, upper in (LOOP_GRAB_LEFT, LOOP_GRAB_RIGHT):
                segments = range(lower, min(upper, len(midpoints)))
                outward = midpoints[segments].mean(axis=0) - knot
                outward[2] = 0.0
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=SETTLE_END,
                        end_time=TIGHTEN_LIFT_END,
                        direction=(0.0, 0.0, 1.0),
                        magnitude=TIGHTEN_LIFT_FORCE,
                        ramp_up=TIGHTEN_LIFT_RAMP,
                        ramp_down=TIGHTEN_LIFT_RAMP,
                    )
                )
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=SETTLE_END,
                        end_time=RELEASE_TIME,
                        direction=tuple(outward),
                        magnitude=TIGHTEN_FORCE,
                        ramp_up=TIGHTEN_FORCE_RAMP,
                        ramp_down=RELEASE_TIME - TIGHTEN_END,
                    )
                )
        for world in range(self.num_envs):
            for (lower, upper), (start, pull_end), force in (
                (TAIL_GRAB_FIRST, (UNTIE_FIRST_START, UNTIE_FIRST_END), UNTIE_FORCE_FIRST),
                (TAIL_GRAB_SECOND, (UNTIE_SECOND_START, UNTIE_SECOND_END), UNTIE_FORCE_SECOND),
            ):
                segments = range(lower, min(upper, len(midpoints)))
                side = 1.0 if midpoints[segments].mean(axis=0)[0] >= knot[0] else -1.0
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=float(start),
                        end_time=float(pull_end + UNTIE_HOLD + UNTIE_RELAX),
                        direction=(side, -0.35, 0.15),
                        magnitude=force,
                        ramp_up=UNTIE_FORCE_RAMP,
                        ramp_down=UNTIE_RELAX,
                    )
                )
            segments = range(*LOOP_GRAB_LEFT)
            outward = midpoints[segments].mean(axis=0) - knot
            horizontal_norm = np.linalg.norm(outward[:2])
            outward[2] = -EXTRACT_DOWNWARD_BIAS * horizontal_norm
            moves.append(
                BodyForceMove(
                    body_indices=self._bodies_for_segments(world, segments),
                    start_time=EXTRACT_START,
                    end_time=EXTRACT_END + UNTIE_HOLD + UNTIE_RELAX,
                    direction=tuple(outward),
                    magnitude=EXTRACT_FORCE,
                    ramp_up=UNTIE_FORCE_RAMP,
                    ramp_down=UNTIE_RELAX,
                )
            )

        return moves

    def _bodies_for_segments(self, world: int, segments: range) -> tuple[int, ...]:
        """Map full-centerline segment indices to simulated bodies in one world."""
        omitted_segments = PINNED_LAST - PINNED_FIRST - 1
        bodies = self.cable_bodies[world]
        return tuple(bodies[segment if segment <= PINNED_FIRST else segment - omitted_segments] for segment in segments)


def create_scene_cfg(
    centerline: np.ndarray,
    cable_radius: float,
    authored_segment_length: float,
    randomize: bool = False,
) -> InteractiveSceneCfg:
    """Create the replicated three-part shoelace scene.

    Args:
        centerline: Resampled cable centerline [m], shape [N, 3].
        cable_radius: Authored cable radius [m].
        authored_segment_length: Mean source-asset segment length [m].
        randomize: Whether to create per-environment writable shoe materials.

    Returns:
        Scene configuration for isolated Newton worlds.
    """
    size = ground_size(args_cli.num_envs, ENV_SPACING)
    cable_normals = parallel_transport_normals(centerline)
    left_centerline = centerline[: PINNED_FIRST + 2]
    right_centerline = centerline[PINNED_LAST:]

    def env_asset(prim_suffix: str, spawn: sim_utils.SpawnerCfg) -> AssetBaseCfg:
        """Create an environment-scoped static asset configuration."""
        return AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}", spawn=spawn)

    shoe_material_cfg = (
        VisualMaterialCfg(
            prim_path="{ENV_REGEX_NS}/Shoe/Visual/material",
            spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.75),
            channels=("color",),
        )
        if randomize
        else None
    )
    shoe_material_path = "./material" if randomize else "/World/ShoeMaterials/shoes"

    @configclass
    class ShoelaceSceneCfg(InteractiveSceneCfg):
        """Replicated shoe and cable assets with shared ground and lighting."""

        shoe_material = shoe_material_cfg
        shoe = shoe_asset_cfg()
        shoe_collider = shoe_collider_asset_cfg()
        shoe_visual = shoe_visual_asset_cfg(shoe_material_path)
        tongue_upper = tongue_upper_asset_cfg()
        shoelace_left = env_asset(
            "ShoelaceLeft",
            cable_spawn_cfg(
                left_centerline,
                cable_normals[: PINNED_FIRST + 2],
                cable_radius,
                authored_segment_length,
            ),
        )
        shoelace_pinned = env_asset("ShoelacePinned", pinned_shoelace_spawner(centerline, cable_radius))
        shoelace_right = env_asset(
            "ShoelaceRight",
            cable_spawn_cfg(
                right_centerline,
                cable_normals[PINNED_LAST:],
                cable_radius,
                authored_segment_length,
            ),
        )
        ground = ground_asset_cfg(size)
        light = light_asset_cfg()

    return ShoelaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=ENV_SPACING, replicate_physics=True)


def run_simulator(sim: sim_utils.SimulationContext, controller: ShoelaceController) -> None:
    """Restore the authored knot and run the requested force-controlled sequence.

    Args:
        sim: Initialized simulation context.
        controller: Configured shoelace controller.
    """
    controller.restore_authored_state()
    for _ in range(args_cli.max_steps):
        if not sim.is_headless_or_exist_active_visualizer():
            break
        sim.step(render=False)
        if sim.is_rendering:
            sim.render()


def main() -> None:
    """Launch Isaac Lab with Newton VBD and run the shoelace demo."""
    centerline, cable_radius, authored_segment_length = load_shoelace()
    shoe_colors = cable_colors = cable_radii = None
    if args_cli.randomize:
        shoe_colors, cable_colors, cable_radii = _sample_randomization(
            args_cli.num_envs, cable_radius, args_cli.randomization_seed
        )
    controller = ShoelaceController(
        centerline,
        cable_radius,
        args_cli.num_envs,
        cable_radii,
        cable_colors,
        cable_inertia_regularization=args_cli.cable_inertia_regularization,
    )

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, VBDSolverCfg):
            raise TypeError("This demo requires the newton_vbd physics preset")
        physics_cfg.num_substeps = SIM_SUBSTEPS
        physics_cfg.collision_decimation = 2
        physics_cfg.default_shape_cfg = NewtonShapeCfg(
            gap=CONTACT_GAP,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=LACE_MU,
        )
        # The default explicit broad phase consumes the precomputed model.shape_contact_pairs.
        physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(
            rigid_contact_max=CONTACTS_PER_ENV * args_cli.num_envs,
            max_triangle_pairs=max(MIN_TRIANGLE_PAIRS, TRIANGLE_PAIRS_PER_ENV * args_cli.num_envs),
        )
        physics_cfg.solver_cfg.iterations = SIM_ITERATIONS
        physics_cfg.solver_cfg.rigid_contact_hard = True
        physics_cfg.solver_cfg.rigid_avbd_alpha = 0.0
        physics_cfg.solver_cfg.rigid_body_contact_buffer_size = VBD_CONTACT_BUFFER

        sim_cfg = sim_utils.SimulationCfg(
            dt=FRAME_DT,
            device=args_cli.device,
            gravity=(0.0, 0.0, 0.0 if args_cli.disable_gravity else GRAVITY),
            physics=physics_cfg,
            visualizer_cfgs=[
                NewtonGLVisualizerCfg(
                    update_frequency=8,
                    enable_shadows=False,
                )
            ],
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        controller.register()

        install_shoe_material_reference()
        scene = InteractiveScene(
            create_scene_cfg(centerline, cable_radius, authored_segment_length, args_cli.randomize)
        )
        camera_eye, camera_target = CAMERA_EYE, CAMERA_TARGET
        if args_cli.num_envs > 1:
            extent = ENV_SPACING * math.ceil(math.sqrt(args_cli.num_envs))
            camera_eye = (0.8 * extent, -0.8 * extent, 0.7 * extent)
            camera_target = (0.0, 0.0, CAMERA_TARGET[2])
        sim.set_camera_view(eye=camera_eye, target=camera_target)

        sim.reset()
        if shoe_colors is not None:
            material = scene["shoe_material"]
            if not isinstance(material, VisualMaterial):
                raise TypeError("Randomized shoe material did not initialize as a VisualMaterial")
            VisualMaterial.write_channels(
                [material],
                {"color": torch.from_numpy(shoe_colors).unsqueeze(0)},
            )
            print(
                f"[INFO]: Randomized shoe/cable colors in {args_cli.num_envs} environments with seed "
                f"{args_cli.randomization_seed}; cable radius "
                f"{1.0e3 * cable_radii.min():.3f}-{1.0e3 * cable_radii.max():.3f} mm."
            )
        print(
            f"[INFO]: Shoelace ready: {args_cli.num_envs} worlds, "
            f"{len(centerline) - 1} visual segments/world, "
            f"{len(controller.cable_bodies[0])} VBD bodies/world, "
            f"anchors [{PINNED_FIRST}, {PINNED_LAST}]."
        )
        run_simulator(sim, controller)


if __name__ == "__main__":
    main()
