"""Screen Capture Module"""

from PIL import Image
from typing import Optional, Tuple
import mss
import mss.tools


class ScreenCapture:
    """Capture screenshots from display"""

    def __init__(self):
        self.sct = mss.mss()

    def capture_full(self, monitor: int = 0) -> Image.Image:
        """Capture full screen

        Args:
            monitor: Monitor index (0 = all monitors, 1 = first, 2 = second, etc.)
        """
        mon = self.sct.monitors[monitor]
        screenshot = self.sct.grab(mon)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int
    ) -> Image.Image:
        """Capture a specific region"""
        region = {"left": left, "top": top, "width": width, "height": height}
        screenshot = self.sct.grab(region)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def capture_window(self, title: str) -> Optional[Image.Image]:
        """Capture a specific window by title (Linux/X11)"""
        try:
            import subprocess
            # Get window ID
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True,
                text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            window_id = result.stdout.strip().split()[0]

            # Get window geometry
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return None

            # Parse geometry
            geo = {}
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, val = line.split("=")
                    geo[key] = int(val)

            return self.capture_region(
                geo["X"], geo["Y"], geo["WIDTH"], geo["HEIGHT"]
            )
        except Exception:
            return None

    def list_monitors(self) -> list:
        """List available monitors"""
        return [
            {
                "index": i,
                "width": m["width"],
                "height": m["height"],
                "left": m["left"],
                "top": m["top"],
            }
            for i, m in enumerate(self.sct.monitors)
        ]

    def save(self, image: Image.Image, path: str):
        """Save screenshot to file"""
        image.save(path)
