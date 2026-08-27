# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable open-loop body-force scheduling for Newton demos."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton import State


@wp.kernel
def _apply_body_forces(
    body_indices: wp.array[wp.int32],
    body_offsets: wp.array[wp.int32],
    body_counts: wp.array[wp.int32],
    start_times: wp.array[wp.float32],
    end_times: wp.array[wp.float32],
    directions: wp.array[wp.vec3],
    magnitudes: wp.array[wp.float32],
    ramp_up: wp.array[wp.float32],
    ramp_down: wp.array[wp.float32],
    elapsed: wp.array[wp.float32],
    sim_dt: float,
    body_f: wp.array[wp.spatial_vector],
):
    move = wp.tid()
    time = elapsed[0] + sim_dt
    if time < start_times[move] or time > end_times[move]:
        return

    up_u = wp.clamp((time - start_times[move]) / wp.max(ramp_up[move], 1.0e-6), 0.0, 1.0)
    down_u = wp.clamp((end_times[move] - time) / wp.max(ramp_down[move], 1.0e-6), 0.0, 1.0)
    up_scale = up_u * up_u * (3.0 - 2.0 * up_u)
    down_scale = down_u * down_u * (3.0 - 2.0 * down_u)
    force = magnitudes[move] * wp.min(up_scale, down_scale) * directions[move] / float(body_counts[move])
    offset = body_offsets[move]
    for index in range(body_counts[move]):
        body_f[body_indices[offset + index]] += wp.spatial_vector(force, wp.vec3())


@wp.kernel
def _advance_elapsed(elapsed: wp.array[wp.float32], sim_dt: float, limit: float):
    elapsed[0] = wp.min(elapsed[0] + sim_dt, limit)


@dataclass(frozen=True)
class BodyForceMove:
    """One smooth force profile shared by a group of Newton bodies.

    Attributes:
        body_indices: Newton body indices that share the force.
        start_time: Profile start time [s].
        end_time: Profile end time [s].
        direction: Force direction in world coordinates; normalized during schedule construction.
        magnitude: Peak total force across the body group [N].
        ramp_up: Smooth engagement duration [s].
        ramp_down: Smooth release duration [s].
    """

    body_indices: tuple[int, ...]
    start_time: float
    end_time: float
    direction: tuple[float, float, float]
    magnitude: float
    ramp_up: float
    ramp_down: float


class OpenLoopBodyForceSchedule:
    """Apply smooth, time-indexed force profiles before every Newton solver substep."""

    def __init__(self, moves: Sequence[BodyForceMove], device: str, sim_dt: float):
        if not moves:
            raise ValueError("An open-loop force schedule requires at least one move")
        if not math.isfinite(sim_dt) or sim_dt <= 0.0:
            raise ValueError(f"sim_dt must be finite and positive, got {sim_dt}")

        body_indices: list[int] = []
        body_offsets: list[int] = []
        directions: list[tuple[float, float, float]] = []
        for move in moves:
            self._validate_move(move)
            body_offsets.append(len(body_indices))
            body_indices.extend(move.body_indices)
            norm = math.sqrt(sum(component * component for component in move.direction))
            directions.append(tuple(component / norm for component in move.direction))

        self._move_count = len(moves)
        self._sim_dt = sim_dt
        self._duration = max(move.end_time for move in moves)
        self._body_indices = wp.array(body_indices, dtype=wp.int32, device=device)
        self._body_offsets = wp.array(body_offsets, dtype=wp.int32, device=device)
        self._body_counts = wp.array([len(move.body_indices) for move in moves], dtype=wp.int32, device=device)
        self._start_times = wp.array([move.start_time for move in moves], dtype=wp.float32, device=device)
        self._end_times = wp.array([move.end_time for move in moves], dtype=wp.float32, device=device)
        self._directions = wp.array(directions, dtype=wp.vec3, device=device)
        self._magnitudes = wp.array([move.magnitude for move in moves], dtype=wp.float32, device=device)
        self._ramp_up = wp.array([move.ramp_up for move in moves], dtype=wp.float32, device=device)
        self._ramp_down = wp.array([move.ramp_down for move in moves], dtype=wp.float32, device=device)
        self._elapsed = wp.zeros(1, dtype=wp.float32, device=device)

    def reset(self) -> None:
        """Reset the schedule clock to zero."""
        self._elapsed.zero_()

    def apply(self, state: State) -> None:
        """Add the scheduled forces to one Newton state."""
        wp.launch(
            _apply_body_forces,
            dim=self._move_count,
            inputs=[
                self._body_indices,
                self._body_offsets,
                self._body_counts,
                self._start_times,
                self._end_times,
                self._directions,
                self._magnitudes,
                self._ramp_up,
                self._ramp_down,
                self._elapsed,
                self._sim_dt,
                state.body_f,
            ],
            device=state.body_f.device,
        )
        wp.launch(
            _advance_elapsed,
            dim=1,
            inputs=[self._elapsed, self._sim_dt, self._duration],
            device=state.body_f.device,
        )

    @staticmethod
    def _validate_move(move: BodyForceMove) -> None:
        if not move.body_indices:
            raise ValueError("Each force move requires at least one body index")
        values = (move.start_time, move.end_time, move.magnitude, move.ramp_up, move.ramp_down, *move.direction)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Force move parameters must be finite")
        if move.end_time < move.start_time:
            raise ValueError("Force move end_time must not precede start_time")
        if move.magnitude < 0.0 or move.ramp_up < 0.0 or move.ramp_down < 0.0:
            raise ValueError("Force magnitude and ramp durations must be nonnegative")
        if math.sqrt(sum(component * component for component in move.direction)) <= 1.0e-12:
            raise ValueError("Force move direction must be nonzero")
