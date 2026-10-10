# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenGL viewport mosaics for policy playback recording."""

from __future__ import annotations

import math

import numpy as np
from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg
from PIL import Image, ImageDraw, ImageFont

from isaaclab.utils.configclass import configclass


@configclass
class _NewtonGLVideoGridCfg(NewtonGLVisualizerCfg):
    """Private playback configuration; each tile uses the OpenGL framebuffer."""

    class_type: str = "isaaclab_rl.entrypoints._newton_gl_video:_NewtonGLVideoGrid"
    headless: bool = True
    streaming_view: bool = False
    enable_live_plots: bool = False
    enable_picking: bool = False
    enable_markers: bool = False
    window_width: int = 960
    window_height: int = 540
    focal_length: float = 24.0
    grid_num_envs: int = 16
    grid_eye: tuple[float, float, float] = (0.0, -0.8, 0.85)
    """Overview eye offset from each environment origin [m]."""
    grid_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Overview look-at offset from each environment origin [m]."""
    closeup_env: int | None = None
    closeup_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at offset from the closeup environment origin [m]."""
    closeup_eye: tuple[float, float, float] = (0.0, -0.35, 0.525)
    """Closeup camera offset from the look-at target in world axes [m]."""


class _NewtonGLVideoGrid(NewtonGLVisualizer):
    """Render synchronized views of one simulation state using Newton's GL viewer."""

    cfg: _NewtonGLVideoGridCfg

    def render_rgb_array(self) -> np.ndarray:
        """Return the GL mosaic, optionally replacing the bottom-right 2 by 2 cells."""
        if self._viewer is None or self._scene_data_provider is None:
            raise RuntimeError("The GL video grid is not initialized.")
        scene = self._scene_data_provider.get_interactive_scene()
        origins = scene.env_origins
        origins = getattr(origins, "torch", origins).cpu().numpy()
        columns = math.ceil(math.sqrt(self.cfg.grid_num_envs))
        rows = math.ceil(self.cfg.grid_num_envs / columns)
        tile_width, tile_height = 1920 // columns, 1080 // rows
        canvas = Image.new("RGB", (1920, 1080), (21, 27, 34))
        previous_pose = self._last_camera_pose
        previous_far = self._viewer.camera.far
        try:
            first_view = True
            for env_id in range(self.cfg.grid_num_envs):
                row, column = divmod(env_id, columns)
                if self.cfg.closeup_env is not None and row >= rows - 2 and column >= columns - 2:
                    continue
                eye = origins[env_id] + np.asarray(self.cfg.grid_eye)
                lookat = origins[env_id] + np.asarray(self.cfg.grid_lookat)
                frame = self._render_view(eye, lookat, update_state=first_view)
                first_view = False
                tile = self._label_frame(frame, (tile_width, tile_height), f"ENV {env_id:02d}")
                canvas.paste(tile, (column * tile_width, row * tile_height))
            if self.cfg.closeup_env is not None:
                env_id = self.cfg.closeup_env
                target = origins[env_id] + np.asarray(self.cfg.closeup_offset)
                label = f"CONTACT DETAIL  /  ENV {env_id:02d}"
                frame = self._render_view(target + np.asarray(self.cfg.closeup_eye), target, update_state=first_view)
                tile = self._label_frame(frame, (2 * tile_width, 2 * tile_height), label, closeup=True)
                canvas.paste(tile, ((columns - 2) * tile_width, (rows - 2) * tile_height))
        finally:
            self._viewer.camera.far = previous_far
            if previous_pose is not None:
                self._apply_camera_pose(previous_pose)
        return np.asarray(canvas)

    def _render_view(self, eye: np.ndarray, lookat: np.ndarray, *, update_state: bool) -> np.ndarray:
        """Upload geometry once per mosaic, then render each camera with the same GL materials."""
        self._apply_camera_pose((tuple(eye), tuple(lookat)))
        # Clip distant environment copies while retaining geometry around the target.
        self._viewer.camera.far = float(np.linalg.norm(eye - lookat)) + 0.6
        if update_state:
            return super().render_rgb_array()
        viewer = self._viewer
        viewer.renderer.render(viewer.camera, viewer.objects, viewer.lines, viewer.wireframe_shapes, viewer.arrows)
        return viewer.get_frame().numpy()

    @staticmethod
    def _label_frame(frame: np.ndarray, size: tuple[int, int], label: str, *, closeup: bool = False) -> Image.Image:
        """Keep labels and a thin separator outside the contact region."""
        tile = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        color = (107, 224, 197) if closeup else (211, 220, 227)
        font = ImageFont.load_default(size=18 if closeup else 14)
        label_width = int(draw.textlength(label, font=font)) + 24
        draw.rectangle((0, 0, label_width, 34 if closeup else 28), fill=(21, 27, 34))
        draw.text((12, 8), label, fill=color, font=font)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=color if closeup else (21, 27, 34), width=3)
        return tile
