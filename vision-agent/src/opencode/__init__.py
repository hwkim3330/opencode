"""OpenCode - Vision-based AI Agent"""

__version__ = "0.1.0"

from .agent.core import VisionAgent, create_agent
from .vision.capture import ScreenCapture
from .vision.model import VisionModel
from .tools.actions import UIActions, Shortcuts

__all__ = [
    "VisionAgent",
    "create_agent",
    "ScreenCapture",
    "VisionModel",
    "UIActions",
    "Shortcuts",
]
