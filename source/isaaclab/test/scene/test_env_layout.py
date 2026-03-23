# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`EnvLayout` and combo space utilities.

No Isaac Sim / USD / PhysX dependency required.
"""

import pytest
import torch

from isaaclab.cloner.cloner_strategies import random as random_strategy
from isaaclab.cloner.cloner_strategies import sequential as sequential_strategy
from isaaclab.scene.clone_cfg import CloneCfg, InclusionSet
from isaaclab.scene.env_layout import EnvLayout, filter_to_group, get_env_ids, to_global, to_local

# ===========================================================================
# EnvLayout: registration & basic queries
# ===========================================================================


class TestRegistration:
    def test_empty_layout_is_homogeneous(self):
        layout = EnvLayout(24, "cpu")
        assert not layout.group_names  # empty tuple is falsy
        assert layout.num_envs == 24
        assert layout.group_names == ()

    def test_register_creates_group(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.tensor([0, 1, 2, 3]))
        assert layout.group_names  # non-empty
        assert "lift" in layout.group_names
        assert layout["lift"].count == 4
        assert layout["lift"].env_ids.tolist() == [0, 1, 2, 3]

    def test_register_out_of_range_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="out of range"):
            layout.register("bad", torch.tensor([0, 1, 100]))

    def test_register_negative_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="out of range"):
            layout.register("bad", torch.tensor([-1, 0, 1]))

    def test_register_duplicates_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="duplicates"):
            layout.register("bad", torch.tensor([0, 1, 1, 2]))

    def test_re_register_overwrites(self):
        layout = EnvLayout(10, "cpu")
        layout.register("g", torch.tensor([0, 1]))
        layout.register("g", torch.tensor([3, 4, 5]))
        assert layout["g"].env_ids.tolist() == [3, 4, 5]
        assert layout["g"].count == 3


# ===========================================================================
# GroupView: env_ids and count
# ===========================================================================


class TestGroupViewBasics:
    def test_env_ids_returns_long_tensor(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", torch.tensor([2, 5]))
        t = layout["g"].env_ids
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.long
        assert t.tolist() == [2, 5]

    def test_group_view_consistent(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", torch.tensor([2, 5]))
        v1 = layout["g"]
        v2 = layout["g"]
        # Views are equal (same indices), but not necessarily same object
        assert v1.write.tolist() == v2.write.tolist()
        assert v1.count == v2.count

    def test_homogeneous_group_view(self):
        layout = EnvLayout(4, "cpu")
        view = layout.group_view(None)
        assert view.write == slice(None)
        assert view.read == slice(None)


# ===========================================================================
# GroupView: write index (replaces env_slice)
# ===========================================================================


class TestWriteIndex:
    def test_contiguous_returns_slice(self):
        layout = EnvLayout(24, "cpu")
        layout.register("stack", torch.arange(8, 16))
        s = layout["stack"].write
        assert isinstance(s, slice)
        assert s == slice(8, 16)

    def test_sparse_returns_tensor(self):
        layout = EnvLayout(24, "cpu")
        layout.register("sparse", torch.tensor([0, 2, 5, 10]))
        s = layout["sparse"].write
        assert isinstance(s, torch.Tensor)
        assert s.tolist() == [0, 2, 5, 10]

    def test_single_element(self):
        layout = EnvLayout(8, "cpu")
        layout.register("one", torch.tensor([3]))
        s = layout["one"].write
        assert isinstance(s, slice)
        assert s == slice(3, 4)


# ===========================================================================
# GroupView: to_local (replaces global_to_local)
# ===========================================================================


class TestToLocal:
    def test_exact_match(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.tensor([4, 7, 10]))
        result = layout["lift"].to_local(torch.tensor([4, 7, 10]))
        assert result.tolist() == [0, 1, 2]

    def test_drops_non_matching(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.tensor([4, 7, 10]))
        result = layout["lift"].to_local(torch.tensor([0, 4, 5, 7, 100]))
        assert result.tolist() == [0, 1]

    def test_empty_when_no_match(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.tensor([4, 7, 10]))
        result = layout["lift"].to_local(torch.tensor([0, 1, 2, 3]))
        assert result.numel() == 0

    def test_preserves_order(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", torch.tensor([10, 20]))
        result = layout["g"].to_local(torch.tensor([20, 10]))
        assert result.tolist() == [1, 0]


# ===========================================================================
# GroupView: to_global (replaces local_to_global)
# ===========================================================================


class TestToGlobal:
    def test_maps_back_correctly(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", torch.tensor([10, 15, 20]))
        result = layout["g"].to_global(torch.tensor([0, 1, 2]))
        assert result.tolist() == [10, 15, 20]

    def test_roundtrip_with_to_local(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", torch.tensor([4, 8, 12]))
        global_ids = torch.tensor([4, 12])
        local = layout["g"].to_local(global_ids)
        recovered = layout["g"].to_global(local)
        assert recovered.tolist() == [4, 12]


# ===========================================================================
# GroupView: filter (replaces filter_and_split)
# ===========================================================================


class TestFilter:
    def test_heterogeneous(self):
        layout = EnvLayout(24, "cpu")
        layout.register("reach", torch.arange(16, 24))
        ids = torch.tensor([2, 5, 18, 20])
        local, glob = layout["reach"].filter(ids)
        assert local.tolist() == [2, 4]
        assert glob.tolist() == [18, 20]

    def test_no_match_returns_empty(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", torch.tensor([10, 11, 12]))
        local, glob = layout["g"].filter(torch.tensor([0, 1, 2]))
        assert local.numel() == 0
        assert glob.numel() == 0


# ===========================================================================
# EnvLayout: entity registry
# ===========================================================================


class TestEntityRegistry:
    def test_register_asset_and_lookup(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", torch.tensor([0, 1, 2]))
        layout.register_asset("robot_a", "lift")
        assert layout.assets.get("robot_a") == ("lift",)
        assert layout.assets.get("unknown") is None

    def test_register_term_and_lookup(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", torch.tensor([0, 1, 2]))
        layout.register_term("joint_pos", "lift")
        assert layout.terms.get("joint_pos") == "lift"
        assert layout.terms.get("unknown") is None

    def test_asset_groups_shared(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(0, 8))
        layout.register("reach", torch.arange(16, 24))
        layout.register_asset("robot", "lift")
        layout.register_asset("table", "lift")
        layout.register_asset("table", "reach")
        # robot is exclusive to lift
        assert layout.assets.get("robot") == ("lift",)
        # table is shared between lift and reach
        assert layout.assets.get("table") == ("lift", "reach")


# ===========================================================================
# EnvLayout: repr
# ===========================================================================


class TestRepr:
    def test_homogeneous_repr(self):
        layout = EnvLayout(8, "cpu")
        r = repr(layout)
        assert "homogeneous" in r
        assert "8" in r

    def test_heterogeneous_repr(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.tensor([0, 1, 2]))
        r = repr(layout)
        assert "lift" in r
        assert "3 envs" in r


# ===========================================================================
# Strategy-based assignment
# ===========================================================================


def _weights_and_names(cfg: CloneCfg) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Helper: extract weights and group names from CloneCfg."""
    group_names = tuple(cfg.clone_groups.keys())
    weights = torch.tensor([cfg.clone_groups[n].weight for n in group_names])
    return weights, group_names


class TestStrategyAssignment:
    """Tests for strategy-based env assignment."""

    def test_sequential_assigns_contiguous_blocks(self):
        weights = torch.tensor([1.0, 1.0, 1.0])
        assignment = sequential_strategy(weights, 12, "cpu")
        assert assignment.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_random_assigns_shuffled_but_correct_counts(self):
        weights = torch.tensor([1.0, 1.0, 1.0])
        assignment = random_strategy(weights, 24, "cpu")
        counts = torch.bincount(assignment, minlength=3)
        assert counts.tolist() == [8, 8, 8]

    def test_weighted_assignment_respects_weights(self):
        weights = torch.tensor([2.0, 1.0])
        assignment = sequential_strategy(weights, 12, "cpu")
        counts = torch.bincount(assignment, minlength=2)
        assert counts[0].item() == 8
        assert counts[1].item() == 4

    def test_apply_assignment_to_layout(self):
        layout = EnvLayout(12, "cpu")
        cfg = CloneCfg(
            clone_groups={
                "a": InclusionSet(assets=["asset_a"], weight=1),
                "b": InclusionSet(assets=["asset_b"], weight=1),
            }
        )
        layout.apply_clone_cfg(cfg)

        weights, group_names = _weights_and_names(cfg)
        assignment = sequential_strategy(weights, 12, "cpu")
        layout.apply_assignment(assignment, group_names)

        assert layout["a"].env_ids.tolist() == [0, 1, 2, 3, 4, 5]
        assert layout["b"].env_ids.tolist() == [6, 7, 8, 9, 10, 11]

    def test_random_assignment_produces_interleaved_groups(self):
        layout = EnvLayout(24, "cpu")
        cfg = CloneCfg(
            clone_groups={
                "x": InclusionSet(assets=["ax"], weight=1),
                "y": InclusionSet(assets=["ay"], weight=1),
                "z": InclusionSet(assets=["az"], weight=1),
            }
        )
        layout.apply_clone_cfg(cfg)

        weights, group_names = _weights_and_names(cfg)
        assignment = random_strategy(weights, 24, "cpu")
        layout.apply_assignment(assignment, group_names)

        for name in group_names:
            assert layout[name].count == 8


# ===========================================================================
# CloneCfg: apply_clone_cfg + apply_assignment
# ===========================================================================


class TestCloneCfgPartitioning:
    """Tests for :meth:`EnvLayout.apply_clone_cfg` and :meth:`apply_assignment`."""

    def _make_cfg(self, groups: dict[str, tuple[list[str], int]]):
        return CloneCfg(
            clone_groups={name: InclusionSet(assets=assets, weight=weight) for name, (assets, weight) in groups.items()}
        )

    def _apply_full(self, layout: EnvLayout, cfg: CloneCfg):
        layout.apply_clone_cfg(cfg)
        weights, group_names = _weights_and_names(cfg)
        assignment = sequential_strategy(weights, layout.num_envs, "cpu")
        layout.apply_assignment(assignment, group_names)

    def test_apply_clone_cfg_equal_weights(self):
        layout = EnvLayout(24, "cpu")
        cfg = self._make_cfg(
            {
                "lift": (["lift_table", "lift_obj"], 1),
                "cabinet": (["cabinet"], 1),
                "reach": (["reach_table"], 1),
            }
        )
        self._apply_full(layout, cfg)
        assert layout["lift"].count == 8
        assert layout["cabinet"].count == 8
        assert layout["reach"].count == 8

    def test_apply_clone_cfg_weighted(self):
        layout = EnvLayout(24, "cpu")
        cfg = self._make_cfg(
            {
                "a": (["asset_a"], 1),
                "b": (["asset_b"], 2),
                "c": (["asset_c"], 1),
            }
        )
        self._apply_full(layout, cfg)
        assert layout["a"].count == 6
        assert layout["b"].count == 12
        assert layout["c"].count == 6

    def test_apply_clone_cfg_registers_assets(self):
        layout = EnvLayout(12, "cpu")
        cfg = self._make_cfg({"grp": (["robot", "table"], 1)})
        self._apply_full(layout, cfg)
        assert layout.assets.get("robot") == ("grp",)
        assert layout.assets.get("table") == ("grp",)

    def test_apply_clone_cfg_shared_asset(self):
        layout = EnvLayout(24, "cpu")
        cfg = self._make_cfg(
            {
                "lift": (["table", "lift_obj"], 1),
                "cabinet": (["cabinet"], 1),
                "reach": (["table"], 1),
            }
        )
        self._apply_full(layout, cfg)
        assert layout.assets.get("table") == ("lift", "reach")
        assert layout.assets.get("cabinet") == ("cabinet",)

    def test_apply_clone_cfg_twice_raises(self):
        layout = EnvLayout(12, "cpu")
        cfg = self._make_cfg({"g": (["a"], 1)})
        layout.apply_clone_cfg(cfg)
        with pytest.raises(RuntimeError):
            layout.apply_clone_cfg(cfg)

    def test_apply_clone_cfg_full_coverage(self):
        layout = EnvLayout(24, "cpu")
        cfg = self._make_cfg(
            {
                "lift": (["a"], 1),
                "stack": (["b"], 1),
                "reach": (["c"], 1),
            }
        )
        self._apply_full(layout, cfg)
        all_ids = set()
        for g in layout.group_names:
            all_ids.update(layout[g].env_ids.tolist())
        assert all_ids == set(range(24))


# ===========================================================================
# GroupView: indexer correctness
# ===========================================================================


class TestGroupView:
    """Tests for :meth:`EnvLayout.group_view`."""

    def test_group_view_contiguous_exclusive(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(0, 8))
        layout.register_asset("robot", "lift")
        view = layout.group_view("lift", "robot")
        assert view.write == slice(0, 8)
        assert view.read == slice(None)

    def test_group_view_contiguous_full_asset(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(8, 16))
        view = layout.group_view("lift", "ground")
        assert view.write == slice(8, 16)
        assert view.read == slice(8, 16)

    def test_group_view_shared_asset_contiguous(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(0, 8))
        layout.register("reach", torch.arange(16, 24))
        layout.register_asset("table", "lift")
        layout.register_asset("table", "reach")

        lift_view = layout.group_view("lift", "table")
        assert lift_view.write == slice(0, 8)
        assert lift_view.read == slice(0, 8)

        reach_view = layout.group_view("reach", "table")
        assert reach_view.write == slice(16, 24)
        assert reach_view.read == slice(8, 16)

    def test_group_view_disjoint_returns_tensor(self):
        layout = EnvLayout(24, "cpu")
        layout.register("sparse", torch.tensor([0, 3, 7]))
        layout.register_asset("bot", "sparse")
        view = layout.group_view("sparse", "bot")
        assert isinstance(view.write, torch.Tensor)
        assert view.write.tolist() == [0, 3, 7]
        assert view.read == slice(None)

    def test_group_view_consistent(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", torch.arange(4))
        layout.register_asset("a", "g")
        v1 = layout.group_view("g", "a")
        v2 = layout.group_view("g", "a")
        # Views have same indices, but callers should cache at init
        assert v1.write == v2.write
        assert v1.read == v2.read

    def test_group_view_homogeneous_identity(self):
        layout = EnvLayout(24, "cpu")
        view = layout.group_view(None, "any_asset")
        assert view.write == slice(None)
        assert view.read == slice(None)

    def test_group_view_before_registration_raises(self):
        layout = EnvLayout(24, "cpu")
        with pytest.raises(KeyError, match="unregistered group"):
            layout.group_view("lift", "table")


# ===========================================================================
# GroupView: functional data flow
# ===========================================================================


class TestGroupViewFunctional:
    """End-to-end scatter/gather with GroupView indexers."""

    def test_scatter_gather_roundtrip(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(0, 8))
        layout.register_asset("robot", "lift")
        view = layout.group_view("lift", "robot")

        asset_data = torch.arange(8, dtype=torch.float)
        full = torch.zeros(24, dtype=torch.float)
        full[view.write] = asset_data[view.read]
        assert full[:8].tolist() == list(range(8))
        assert full[8:].sum().item() == 0.0

    def test_shared_asset_scatter(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", torch.arange(0, 8))
        layout.register("reach", torch.arange(16, 24))
        layout.register_asset("table", "lift")
        layout.register_asset("table", "reach")

        asset_data = torch.arange(16, dtype=torch.float)
        full = torch.zeros(24, dtype=torch.float)

        lv = layout.group_view("lift", "table")
        full[lv.write] = asset_data[lv.read]

        rv = layout.group_view("reach", "table")
        full[rv.write] = asset_data[rv.read]

        assert full[:8].tolist() == list(range(8))
        assert full[8:16].tolist() == [0.0] * 8
        assert full[16:24].tolist() == list(range(8, 16))


# ===========================================================================
# Pure functions: unit tests
# ===========================================================================


class TestPureFunctions:
    """Unit tests for pure functions in Layer 1."""

    def test_filter_to_group_contiguous(self):
        from isaaclab.scene.env_layout import GroupLayout

        layout = GroupLayout(offset=4, count=4, slice=slice(4, 8), device="cpu")
        env_ids = torch.tensor([2, 4, 5, 10])
        local, matched = filter_to_group(layout, env_ids)
        assert local.tolist() == [0, 1]
        assert matched.tolist() == [4, 5]

    def test_to_local_sparse(self):
        from isaaclab.scene.env_layout import GroupLayout

        indices = torch.tensor([3, 7, 11])
        layout = GroupLayout(offset=3, count=3, slice=None, indices=indices, device="cpu")
        env_ids = torch.tensor([3, 11, 99])
        result = to_local(layout, env_ids)
        assert result.tolist() == [0, 2]

    def test_to_global_sparse(self):
        from isaaclab.scene.env_layout import GroupLayout

        indices = torch.tensor([3, 7, 11])
        layout = GroupLayout(offset=3, count=3, slice=None, indices=indices, device="cpu")
        local_ids = torch.tensor([0, 2])
        result = to_global(layout, local_ids)
        assert result.tolist() == [3, 11]

    def test_get_env_ids_contiguous(self):
        from isaaclab.scene.env_layout import GroupLayout

        layout = GroupLayout(offset=4, count=4, slice=slice(4, 8), device="cpu")
        result = get_env_ids(layout)
        assert result.tolist() == [4, 5, 6, 7]

    def test_get_env_ids_sparse(self):
        from isaaclab.scene.env_layout import GroupLayout

        indices = torch.tensor([3, 7, 11])
        layout = GroupLayout(offset=3, count=3, slice=None, indices=indices, device="cpu")
        result = get_env_ids(layout)
        assert result.tolist() == [3, 7, 11]


# ===========================================================================
# GroupView: graphability (CUDA)
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGroupViewGraphability:
    """Verify GroupView indexers produce fixed-shape outputs (graph-safe)."""

    def test_write_index_fixed_size(self):
        layout = EnvLayout(24, "cuda:0")
        layout.register("g", torch.tensor([0, 3, 7], device="cuda:0"))
        layout.register_asset("a", "g")
        view = layout.group_view("g", "a")
        if isinstance(view.write, torch.Tensor):
            assert view.write.shape[0] == 3

    def test_read_index_fixed_size(self):
        layout = EnvLayout(24, "cuda:0")
        layout.register("lift", torch.arange(0, 8, device="cuda:0"))
        layout.register("reach", torch.arange(16, 24, device="cuda:0"))
        layout.register_asset("table", "lift")
        layout.register_asset("table", "reach")
        view = layout.group_view("lift", "table")
        if isinstance(view.read, torch.Tensor):
            assert view.read.shape[0] == 8
        elif isinstance(view.read, slice):
            data = torch.zeros(16, device="cuda:0")
            assert data[view.read].shape[0] == 8

    def test_indexing_produces_fixed_shape(self):
        layout = EnvLayout(24, "cuda:0")
        layout.register("g", torch.tensor([0, 3, 7, 10], device="cuda:0"))
        layout.register_asset("a", "g")
        view = layout.group_view("g", "a")

        full_buf = torch.randn(24, 4, device="cuda:0")
        asset_buf = torch.randn(4, 4, device="cuda:0")

        written = full_buf[view.write]
        assert written.shape == (4, 4)

        read = asset_buf[view.read]
        assert read.shape == (4, 4)
