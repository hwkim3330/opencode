"""Vision Agent Core"""

from typing import Optional, Callable
from PIL import Image

from ..vision.model import VisionModel
from ..vision.capture import ScreenCapture
from ..tools.actions import UIActions, Shortcuts


class VisionAgent:
    """AI Agent that can see and interact with the screen"""

    def __init__(self, device: Optional[str] = None):
        self.vision = VisionModel(device)
        self.capture = ScreenCapture()
        self.actions = UIActions()
        self.shortcuts = Shortcuts()
        self._loaded = False

    def load(self):
        """Load the vision model"""
        if not self._loaded:
            self.vision.load()
            self._loaded = True
        return self

    def see(self, monitor: int = 1) -> Image.Image:
        """Capture current screen"""
        return self.capture.capture_full(monitor)

    def describe(self, image: Optional[Image.Image] = None) -> str:
        """Describe what's on screen"""
        if image is None:
            image = self.see()
        return self.vision.describe_screen(image)

    def find(self, element: str, image: Optional[Image.Image] = None) -> str:
        """Find a UI element on screen"""
        if image is None:
            image = self.see()
        return self.vision.find_element(image, element)

    def read_error(self, image: Optional[Image.Image] = None) -> str:
        """Read and explain any error on screen"""
        if image is None:
            image = self.see()
        return self.vision.read_error(image)

    def ocr(self, image: Optional[Image.Image] = None) -> str:
        """Extract text from screen"""
        if image is None:
            image = self.see()
        return self.vision.extract_text(image)

    def ask(self, question: str, image: Optional[Image.Image] = None) -> str:
        """Ask any question about the screen"""
        if image is None:
            image = self.see()
        return self.vision.analyze(image, question)

    def click_at(self, x: int, y: int):
        """Click at coordinates"""
        self.actions.click(x, y)

    def type(self, text: str):
        """Type text"""
        self.actions.type_text(text)

    def type_korean(self, text: str):
        """Type Korean text"""
        self.actions.type_unicode(text)

    def press_key(self, key: str):
        """Press a key"""
        self.actions.press(key)

    def hotkey(self, *keys):
        """Press hotkey combination"""
        self.actions.hotkey(*keys)

    def execute_task(self, task: str, max_steps: int = 10) -> list:
        """
        Execute a task by looking at the screen and taking actions.

        This is an experimental feature - the model will try to understand
        the task, look at the screen, and decide what to do.
        """
        results = []

        prompt = f"""You are a UI automation agent. Your task is: {task}

Look at the screen and decide what action to take.
Respond with ONE of these actions:
- CLICK x y - click at coordinates
- TYPE text - type text
- PRESS key - press a key (enter, tab, escape, etc.)
- HOTKEY key1 key2 - press key combination
- SCROLL amount - scroll (positive=up, negative=down)
- DONE - task is complete
- FAIL reason - cannot complete task

Respond with just the action, nothing else."""

        for step in range(max_steps):
            screen = self.see()
            response = self.vision.analyze(screen, prompt, max_tokens=50)
            response = response.strip().upper()

            results.append({"step": step + 1, "action": response})

            if response.startswith("DONE"):
                break
            elif response.startswith("FAIL"):
                break
            elif response.startswith("CLICK"):
                parts = response.split()
                if len(parts) >= 3:
                    x, y = int(parts[1]), int(parts[2])
                    self.click_at(x, y)
            elif response.startswith("TYPE"):
                text = response[5:].strip()
                self.type(text)
            elif response.startswith("PRESS"):
                key = response[6:].strip().lower()
                self.press_key(key)
            elif response.startswith("HOTKEY"):
                keys = response[7:].strip().lower().split()
                self.hotkey(*keys)
            elif response.startswith("SCROLL"):
                amount = int(response[7:].strip())
                self.actions.scroll(amount)

        return results


def create_agent(device: Optional[str] = None) -> VisionAgent:
    """Create and load a vision agent"""
    agent = VisionAgent(device)
    agent.load()
    return agent
