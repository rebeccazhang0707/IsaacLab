# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch tests for :class:`TemplateClonePlan` with flat prototype structure.

No Isaac Sim / USD / PhysX dependency required.
"""

import torch

from isaaclab.cloner.cloner_cfg import TemplateClonePlan


class TestTemplateClonePlan:
    """Tests for TemplateClonePlan data structure with flat prototype structure."""

    def test_homogeneous_creates_single_partition(self):
        """Homogeneous plan has one partition with all prototypes active."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1")
        dest_paths = ("Robot/", "Table/")
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.zeros(8, dtype=torch.long),
            group_mask=torch.ones((1, 2), dtype=torch.bool),
        )

        assert plan.prototype_paths == protos
        assert plan.dest_paths == dest_paths
        assert plan.partition_assignment.shape == (8,)
        assert torch.equal(plan.partition_assignment, torch.zeros(8, dtype=torch.long))
        assert plan.group_mask.shape == (1, 2)
        assert plan.group_mask.all()

    def test_homogeneous_empty_protos(self):
        """Homogeneous plan handles empty prototypes."""
        plan = TemplateClonePlan(
            prototype_paths=(),
            dest_paths=(),
            partition_assignment=torch.zeros(4, dtype=torch.long),
            group_mask=torch.ones((1, 0), dtype=torch.bool),
        )

        assert plan.prototype_paths == ()
        assert plan.dest_paths == ()
        assert plan.partition_assignment.shape == (4,)
        assert plan.group_mask.shape == (1, 0)

    def test_plan_with_multiple_partitions(self):
        """Plan with multiple partitions directly specifies which prototypes to clone."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1", "/World/template/prototype_2")
        dest_paths = ("Robot/", "Table/", "Object/")
        # Partition 0: envs 0-3 clone prototype_0 and prototype_1
        # Partition 1: envs 4-7 clone prototype_0 and prototype_2
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            group_mask=torch.tensor([[True, True, False], [True, False, True]]),
        )

        assert plan.group_mask.shape == (2, 3)
        # Partition 0: prototype_0 and prototype_1 active
        assert plan.group_mask[0, 0] and plan.group_mask[0, 1] and not plan.group_mask[0, 2]
        # Partition 1: prototype_0 and prototype_2 active
        assert plan.group_mask[1, 0] and not plan.group_mask[1, 1] and plan.group_mask[1, 2]

    def test_dest_paths_for_nested_sensors(self):
        """Destination paths can include nested paths for sensors."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1", "/World/template/prototype_2")
        dest_paths = ("Robot/", "Table/", "Robot/panda_hand/wrist_cam")
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.zeros(4, dtype=torch.long),
            group_mask=torch.ones((1, 3), dtype=torch.bool),
        )

        assert plan.dest_paths[0] == "Robot/"
        assert plan.dest_paths[1] == "Table/"
        assert plan.dest_paths[2] == "Robot/panda_hand/wrist_cam"


class TestCloneMaskingFromPlan:
    """Tests for building clone_masking from TemplateClonePlan."""

    def _build_masking(self, plan: TemplateClonePlan) -> torch.Tensor:
        """Build clone_masking tensor from plan (mirrors clone_from_template logic)."""
        return plan.group_mask[plan.partition_assignment].T

    def test_homogeneous_all_envs_get_all_protos(self):
        """Homogeneous plan assigns all prototypes to all envs."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1")
        dest_paths = ("Robot/", "Table/")
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.zeros(4, dtype=torch.long),
            group_mask=torch.ones((1, 2), dtype=torch.bool),
        )

        masking = self._build_masking(plan)

        assert masking.shape == (2, 4)
        assert masking.all(), "All prototypes should be cloned to all envs"

    def test_partitions_clone_different_protos(self):
        """Different partitions clone different prototypes."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1")
        dest_paths = ("Robot/", "Table/")
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.tensor([0, 0, 1, 1]),
            group_mask=torch.tensor([[True, False], [False, True]]),
        )

        masking = self._build_masking(plan)

        assert masking.shape == (2, 4)
        # Partition 0 (envs 0,1) uses prototype_0 only
        assert masking[0, 0] and masking[0, 1]
        assert not masking[1, 0] and not masking[1, 1]
        # Partition 1 (envs 2,3) uses prototype_1 only
        assert masking[1, 2] and masking[1, 3]
        assert not masking[0, 2] and not masking[0, 3]

    def test_multiple_protos_per_partition(self):
        """Partition can clone multiple prototypes."""
        protos = ("/World/template/prototype_0", "/World/template/prototype_1", "/World/template/prototype_2")
        dest_paths = ("Robot/", "Table/", "Object/")
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.tensor([0, 0, 0, 0]),
            group_mask=torch.tensor([[True, True, False]]),  # Clone proto 0 and 1, not 2
        )

        masking = self._build_masking(plan)

        assert masking[0].all(), "prototype_0 cloned to all envs"
        assert masking[1].all(), "prototype_1 cloned to all envs"
        assert not masking[2].any(), "prototype_2 not cloned"

    def test_three_partitions_asset_variants(self):
        """Three partitions for asset with 3 variants."""
        # Robot is prototype_0, Table is prototype_1, Object variants are prototype_2, prototype_3, prototype_4
        protos = tuple(f"/World/template/prototype_{i}" for i in range(5))
        dest_paths = ("Robot/", "Table/", "Object/", "Object/", "Object/")  # Object variants share dest
        # Partition 0: Robot + Table + Object:banana (proto 2)
        # Partition 1: Robot + Table + Object:apple (proto 3)
        # Partition 2: Robot + Table + Object:orange (proto 4)
        plan = TemplateClonePlan(
            prototype_paths=protos,
            dest_paths=dest_paths,
            partition_assignment=torch.tensor([0, 0, 1, 1, 2, 2]),
            group_mask=torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, False, True, False],
                    [True, True, False, False, True],
                ]
            ),
        )

        masking = self._build_masking(plan)

        # Robot and Table cloned to all envs
        assert masking[0].all(), "Robot (proto_0) cloned to all"
        assert masking[1].all(), "Table (proto_1) cloned to all"
        # Object variants distributed
        assert masking[2, 0] and masking[2, 1], "Object:banana to envs 0-1"
        assert masking[3, 2] and masking[3, 3], "Object:apple to envs 2-3"
        assert masking[4, 4] and masking[4, 5], "Object:orange to envs 4-5"
        # No cross-over
        assert not masking[2, 2] and not masking[2, 3]
        assert not masking[3, 0] and not masking[3, 1]
