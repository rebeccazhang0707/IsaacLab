# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for explicit cable-only proxy inertia in the task and standalone demos."""

import importlib
import warnings
from pathlib import Path
from types import ModuleType

import newton
import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.contrib.shoelace import shoelace_physics
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg


@pytest.fixture(params=["task", "demo"])
def physics(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Exercise the self-contained task and demo physics implementations."""
    if request.param == "task":
        return shoelace_physics
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/demos/shoelace"))
    return importlib.import_module("_newton_shoelace")


@pytest.mark.parametrize("regularization", [0.0, 3.0e-7, 1.0e-6])
def test_cable_inertia_preserves_mass_and_unselected_bodies(physics: ModuleType, regularization: float) -> None:
    """Add isotropic inertia only to selected dynamic bodies and update their inverses."""
    builder = newton.ModelBuilder()
    geometric = 4.0e-11 * np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    cable = builder.add_body(mass=4.0e-5, inertia=wp.mat33(geometric))
    anchor = builder.add_body(mass=0.0, inertia=wp.mat33())
    robot = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
    masses = list(builder.body_mass)
    inverse_masses = list(builder.body_inv_mass)
    before = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3).copy()

    physics.configure_cable_inertia(builder, [cable, anchor], regularization)

    expected = before.copy()
    expected[cable] += np.eye(3, dtype=np.float32) * regularization
    actual = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3)
    np.testing.assert_array_equal(actual, expected)
    assert builder.body_mass == masses
    assert builder.body_inv_mass == inverse_masses
    for index in (cable, robot):
        inverse = np.asarray(builder.body_inv_inertia[index], dtype=np.float32).reshape(3, 3)
        np.testing.assert_allclose(actual[index] @ inverse, np.eye(3), atol=1.0e-6)
    np.testing.assert_array_equal(np.asarray(builder.body_inv_inertia[anchor]), 0.0)


@pytest.mark.parametrize("detailed", [False, True])
def test_proxy_inertia_survives_newton_validation(
    physics: ModuleType, detailed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve explicitly authored proxy inertia without a second Newton correction."""
    monkeypatch.setattr(newton, "use_coord_layout_targets", True)
    regularization = ShoelaceEnvCfg().cable_inertia_regularization
    assert regularization == physics.CABLE_INERTIA_REGULARIZATION == 1.0e-6
    builder = newton.ModelBuilder()
    builder.validate_inertia_detailed = detailed
    cable = builder.add_body(mass=4.0e-5, inertia=wp.mat33(np.diag([4.0e-11, 8.0e-11, 8.0e-11])))
    physics.configure_cable_inertia(builder, [cable], regularization)
    expected = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = builder.finalize(device="cpu")

    assert not caught, [str(warning.message) for warning in caught]
    np.testing.assert_array_equal(model.body_inertia.numpy(), expected)
    np.testing.assert_allclose(expected[cable] @ model.body_inv_inertia.numpy()[cable], np.eye(3), atol=1.0e-6)


@pytest.mark.parametrize("regularization", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_proxy_inertia_is_rejected(physics: ModuleType, regularization: float) -> None:
    """Reject negative and nonfinite regularization before editing body properties."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        physics.configure_cable_inertia(newton.ModelBuilder(), [], regularization)
