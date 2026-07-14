# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv subclass that installs :class:`DgpoLiberoCommandManager`."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from ..mdp.demos.command_manager import DgpoLiberoCommandManager


class DgpoManagerBasedRLEnv(ManagerBasedRLEnv):
    """RL env that expands demo command semantic keys like playground Libero."""

    def load_managers(self):
        self.command_manager = DgpoLiberoCommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # Observation / action managers (parent ManagerBasedEnv path).
        super(ManagerBasedRLEnv, self).load_managers()

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }


# Short-lived alias for gym registration / older docs.
CompatManagerBasedRLEnv = DgpoManagerBasedRLEnv
