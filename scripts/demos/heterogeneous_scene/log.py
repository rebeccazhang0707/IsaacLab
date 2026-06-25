# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Colored console logging for the heterogeneous-scene demo's discovery phase.

The repo has no shared color-logging utility (``isaaclab.cli.utils`` only offers
fixed ``[INFO]``/``[WARNING]`` labels), so these two helpers keep the demo's own
``[skip]``/``[dup ]``/``[note]`` scheme. The one bit worth borrowing is the color guard:
ANSI codes are emitted only to a real terminal and skipped under ``NO_COLOR`` /
``TERM=dumb`` / when piped, so redirected logs stay clean.
"""

from __future__ import annotations

import os
import sys

# ANSI colors: yellow when a task is dropped (skipped / duplicate), dim gray for softer
# notes (a kept task with an asset adjusted), against normal output.
_C_SKIP = "\033[33m"
_C_NOTE = "\033[90m"
_C_RESET = "\033[0m"


def _supports_color() -> bool:
    """Whether stdout is a real terminal that should receive ANSI color codes."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def skip_log(msg: str) -> None:
    """Print a task-dropping discovery message (skip / duplicate) in a distinct color."""
    print(f"{_C_SKIP}{msg}{_C_RESET}" if _supports_color() else msg)


def note_log(msg: str) -> None:
    """Print a soft discovery note (task kept, an asset adjusted) in a dim color."""
    print(f"{_C_NOTE}{msg}{_C_RESET}" if _supports_color() else msg)
