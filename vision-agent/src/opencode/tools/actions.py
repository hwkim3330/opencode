"""UI Automation Actions"""

import pyautogui
import time
from typing import Optional, Tuple


# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small pause between actions


class UIActions:
    """Mouse and keyboard automation"""

    def __init__(self):
        self.screen_size = pyautogui.size()

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """Click at position"""
        pyautogui.click(x, y, button=button, clicks=clicks)

    def double_click(self, x: int, y: int):
        """Double click at position"""
        pyautogui.doubleClick(x, y)

    def right_click(self, x: int, y: int):
        """Right click at position"""
        pyautogui.rightClick(x, y)

    def move_to(self, x: int, y: int, duration: float = 0.2):
        """Move mouse to position"""
        pyautogui.moveTo(x, y, duration=duration)

    def drag_to(self, x: int, y: int, duration: float = 0.5):
        """Drag from current position to target"""
        pyautogui.dragTo(x, y, duration=duration)

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None):
        """Scroll wheel (positive = up, negative = down)"""
        pyautogui.scroll(amount, x, y)

    def type_text(self, text: str, interval: float = 0.02):
        """Type text character by character"""
        pyautogui.typewrite(text, interval=interval)

    def type_unicode(self, text: str):
        """Type unicode text (한글 등)"""
        import subprocess
        subprocess.run(["xdotool", "type", "--clearmodifiers", text])

    def press(self, key: str):
        """Press a single key"""
        pyautogui.press(key)

    def hotkey(self, *keys):
        """Press key combination (e.g., 'ctrl', 'c')"""
        pyautogui.hotkey(*keys)

    def key_down(self, key: str):
        """Hold key down"""
        pyautogui.keyDown(key)

    def key_up(self, key: str):
        """Release key"""
        pyautogui.keyUp(key)

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()

    def locate_on_screen(self, image_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """Find image on screen and return center position"""
        try:
            location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
            return location
        except Exception:
            return None

    def wait_for_image(
        self,
        image_path: str,
        timeout: float = 10.0,
        confidence: float = 0.8
    ) -> Optional[Tuple[int, int]]:
        """Wait for image to appear on screen"""
        start = time.time()
        while time.time() - start < timeout:
            pos = self.locate_on_screen(image_path, confidence)
            if pos:
                return pos
            time.sleep(0.5)
        return None


# Common keyboard shortcuts
class Shortcuts:
    @staticmethod
    def copy():
        pyautogui.hotkey("ctrl", "c")

    @staticmethod
    def paste():
        pyautogui.hotkey("ctrl", "v")

    @staticmethod
    def cut():
        pyautogui.hotkey("ctrl", "x")

    @staticmethod
    def undo():
        pyautogui.hotkey("ctrl", "z")

    @staticmethod
    def redo():
        pyautogui.hotkey("ctrl", "shift", "z")

    @staticmethod
    def save():
        pyautogui.hotkey("ctrl", "s")

    @staticmethod
    def select_all():
        pyautogui.hotkey("ctrl", "a")

    @staticmethod
    def find():
        pyautogui.hotkey("ctrl", "f")

    @staticmethod
    def new_tab():
        pyautogui.hotkey("ctrl", "t")

    @staticmethod
    def close_tab():
        pyautogui.hotkey("ctrl", "w")

    @staticmethod
    def switch_window():
        pyautogui.hotkey("alt", "tab")
