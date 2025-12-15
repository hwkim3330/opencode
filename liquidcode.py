#!/usr/bin/env python3
"""
LiquidCode - LiquidAI LFM2-VL 기반 AI 코딩 어시스턴트 CLI

Claude Code / OpenCode 스타일의 터미널 기반 AI 코딩 도구
LFM2-VL 비전 모델을 사용하여 스크린샷/이미지 분석 지원
(Claude Code의 단점인 비전 기능 없음을 해결)
"""

import os
import sys
import json
import glob
import subprocess
import re
import argparse
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Rich TUI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box

# AI & Vision
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

console = Console()

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """LiquidCode 설정"""
    # LFM2-VL 비전 모델 (Claude Code와 차별화)
    model_name: str = "LiquidAI/LFM2-VL-450M"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens: int = 2048
    temperature: float = 0.7
    working_dir: str = os.getcwd()
    history_file: str = os.path.expanduser("~/.liquidcode_history")
    screenshot_dir: str = "/tmp/liquidcode_screenshots"

CONFIG = Config()

# ============================================================
# Vision Tools - Claude Code에 없는 핵심 기능
# ============================================================

class VisionTools:
    """비전 도구들 - LFM2-VL로 구현"""

    @staticmethod
    def take_screenshot(region: str = "full") -> Tuple[str, Image.Image]:
        """스크린샷 캡처
        Args:
            region: 'full' or 'x,y,w,h' 좌표
        Returns:
            (파일경로, PIL Image)
        """
        try:
            import mss

            os.makedirs(CONFIG.screenshot_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{CONFIG.screenshot_dir}/screenshot_{timestamp}.png"

            with mss.mss() as sct:
                if region == "full":
                    monitor = sct.monitors[0]  # All monitors
                else:
                    try:
                        x, y, w, h = map(int, region.split(','))
                        monitor = {"left": x, "top": y, "width": w, "height": h}
                    except:
                        monitor = sct.monitors[0]

                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                img.save(filepath)

            return filepath, img
        except ImportError:
            # Fallback to scrot
            try:
                os.makedirs(CONFIG.screenshot_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"{CONFIG.screenshot_dir}/screenshot_{timestamp}.png"
                subprocess.run(["scrot", filepath], check=True)
                img = Image.open(filepath)
                return filepath, img
            except Exception as e:
                return f"Error: {e}", None
        except Exception as e:
            return f"Error: {e}", None

    @staticmethod
    def load_image(file_path: str) -> Tuple[str, Image.Image]:
        """이미지 파일 로드"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(CONFIG.working_dir) / path

            if not path.exists():
                return f"Error: File not found: {path}", None

            img = Image.open(path).convert("RGB")
            return str(path), img
        except Exception as e:
            return f"Error loading image: {e}", None


# ============================================================
# Tools - Claude Code 스타일 + Vision
# ============================================================

class Tools:
    """코딩 도구들"""

    vision = VisionTools()

    @staticmethod
    def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """파일 읽기"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(CONFIG.working_dir) / path

            if not path.exists():
                return f"Error: File not found: {path}"

            # 이미지 파일인 경우
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                return f"[IMAGE FILE: {path}] Use analyze_image tool to view this file."

            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Line numbers
            total = len(lines)
            start = offset
            end = min(offset + limit, total)

            result = []
            for i, line in enumerate(lines[start:end], start=start+1):
                result.append(f"{i:6d}│{line.rstrip()}")

            return f"File: {path} ({total} lines)\n" + "\n".join(result)
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """파일 쓰기"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(CONFIG.working_dir) / path

            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def edit_file(file_path: str, old_string: str, new_string: str) -> str:
        """파일 편집 (문자열 대체)"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(CONFIG.working_dir) / path

            if not path.exists():
                return f"Error: File not found: {path}"

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_string not in content:
                return f"Error: String not found in file"

            # 유일성 체크
            count = content.count(old_string)
            if count > 1:
                return f"Error: String found {count} times. Provide more context for unique match."

            new_content = content.replace(old_string, new_string)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def bash(command: str, timeout: int = 60) -> str:
        """Bash 명령 실행"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=CONFIG.working_dir
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

    @staticmethod
    def glob_files(pattern: str, path: str = ".") -> str:
        """파일 패턴 검색"""
        try:
            base = Path(path) if Path(path).is_absolute() else Path(CONFIG.working_dir) / path
            matches = list(base.glob(pattern))

            if not matches:
                return "No files matched the pattern"

            # Sort by modification time
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            result = [f"Found {len(matches)} files:"]
            for m in matches[:50]:  # Limit to 50
                result.append(f"  {m.relative_to(CONFIG.working_dir)}")

            if len(matches) > 50:
                result.append(f"  ... and {len(matches) - 50} more")

            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def grep(pattern: str, path: str = ".", file_type: str = None) -> str:
        """내용 검색 (ripgrep 스타일)"""
        try:
            cmd = ["grep", "-rn", "--color=never"]

            if file_type:
                cmd.extend(["--include", f"*.{file_type}"])

            cmd.extend([pattern, path])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=CONFIG.working_dir
            )

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 50:
                    return "\n".join(lines[:50]) + f"\n... and {len(lines) - 50} more matches"
                return result.stdout.strip()
            return "No matches found"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def list_dir(path: str = ".") -> str:
        """디렉토리 목록"""
        try:
            p = Path(path) if Path(path).is_absolute() else Path(CONFIG.working_dir) / path

            if not p.exists():
                return f"Error: Path not found: {p}"

            if not p.is_dir():
                return f"Error: Not a directory: {p}"

            items = list(p.iterdir())
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            result = [f"Directory: {p}"]
            for item in items[:100]:
                if item.is_dir():
                    result.append(f"  📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    result.append(f"  📄 {item.name} ({size:,} bytes)")

            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# LLM Engine with Vision
# ============================================================

class LLMEngine:
    """LFM2-VL 비전 모델 엔진"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = CONFIG.device
        self.loaded = False

    def load(self):
        """모델 로드"""
        if self.loaded:
            return

        console.print(f"[cyan]Loading {CONFIG.model_name}...[/cyan]")

        try:
            self.processor = AutoProcessor.from_pretrained(
                CONFIG.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                CONFIG.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            self.model.eval()
            self.loaded = True

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                console.print(f"[green]Model loaded (GPU: {mem:.2f} GB)[/green]")
            else:
                console.print("[green]Model loaded (CPU)[/green]")

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise

    def generate(self, prompt: str, image: Image.Image = None, max_tokens: int = None) -> str:
        """텍스트/비전 생성"""
        if not self.loaded:
            self.load()

        max_tokens = max_tokens or CONFIG.max_tokens

        try:
            if image is not None:
                # Vision + Text 모드
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Text only 모드
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=CONFIG.temperature,
                    top_p=0.95,
                )

            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from response if present
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            return f"Error generating response: {e}"


# ============================================================
# Agent with Vision
# ============================================================

@dataclass
class Message:
    role: str  # user, assistant, system, tool
    content: str
    tool_name: str = None
    tool_result: str = None
    image: Image.Image = None

class Agent:
    """LiquidCode 에이전트 (비전 지원)"""

    SYSTEM_PROMPT = """You are LiquidCode, an AI coding assistant with VISION capability running in the terminal.
You can SEE images and screenshots - this is your key advantage over Claude Code.

Available tools:
- read_file(file_path, offset=0, limit=2000): Read a file with line numbers
- write_file(file_path, content): Create or overwrite a file
- edit_file(file_path, old_string, new_string): Replace text in a file
- bash(command): Execute a shell command
- glob_files(pattern, path="."): Find files by pattern
- grep(pattern, path=".", file_type=None): Search file contents
- list_dir(path="."): List directory contents
- screenshot(region="full"): Take a screenshot and analyze it (VISION)
- analyze_image(file_path): Analyze an image file (VISION)

To use a tool, output in this format:
<tool name="tool_name">
<param name="param_name">value</param>
</tool>

VISION CAPABILITIES:
- You can see and analyze screenshots
- You can read and understand images
- Use screenshot() to see what's on screen
- Use analyze_image() to examine image files

After using tools, explain what you did and suggest next steps.
Be concise and helpful. Focus on solving the user's problem.
"""

    def __init__(self):
        self.llm = LLMEngine()
        self.messages: List[Message] = []
        self.tools = Tools()
        self.current_image: Image.Image = None

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))

    def build_prompt(self) -> str:
        """대화 기록을 프롬프트로 변환"""
        parts = [self.SYSTEM_PROMPT, ""]

        for msg in self.messages[-10:]:  # Last 10 messages
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                parts.append(f"Tool ({msg.tool_name}): {msg.tool_result[:1000]}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    def parse_tool_calls(self, response: str) -> List[Dict]:
        """응답에서 도구 호출 파싱 (다양한 형식 지원)"""
        tools = []

        # 형식 1: <tool name="...">...</tool>
        pattern1 = r'<tool name="(\w+)">(.*?)</tool>'
        matches = re.findall(pattern1, response, re.DOTALL)
        for tool_name, params_str in matches:
            params = {}
            param_pattern = r'<param name="(\w+)">(.*?)</param>'
            param_matches = re.findall(param_pattern, params_str, re.DOTALL)
            for param_name, param_value in param_matches:
                params[param_name] = param_value.strip()
            tools.append({"name": tool_name, "params": params})

        # 형식 2: <tool_name>arg</tool_name> (단순 형식)
        simple_tools = ['list_dir', 'read_file', 'bash', 'glob_files', 'grep', 'screenshot', 'analyze_image']
        for tool in simple_tools:
            pattern2 = rf'<{tool}>\s*(.*?)\s*</{tool}>'
            matches = re.findall(pattern2, response, re.DOTALL)
            for arg in matches:
                if tool == 'list_dir':
                    tools.append({"name": tool, "params": {"path": arg.strip() or "."}})
                elif tool == 'read_file':
                    tools.append({"name": tool, "params": {"file_path": arg.strip()}})
                elif tool == 'bash':
                    tools.append({"name": tool, "params": {"command": arg.strip()}})
                elif tool == 'glob_files':
                    tools.append({"name": tool, "params": {"pattern": arg.strip()}})
                elif tool == 'grep':
                    tools.append({"name": tool, "params": {"pattern": arg.strip()}})
                elif tool == 'screenshot':
                    tools.append({"name": tool, "params": {"region": arg.strip() or "full"}})
                elif tool == 'analyze_image':
                    tools.append({"name": tool, "params": {"file_path": arg.strip()}})

        # 형식 3: <tool_name attr="value"/> (self-closing)
        pattern3 = r'<(\w+)\s+([^>]+)/>'
        matches = re.findall(pattern3, response)
        for tool_name, attrs_str in matches:
            if tool_name in simple_tools or tool_name in ['write_file', 'edit_file']:
                params = {}
                attr_pattern = r'(\w+)="([^"]*)"'
                attr_matches = re.findall(attr_pattern, attrs_str)
                for attr_name, attr_value in attr_matches:
                    params[attr_name] = attr_value
                if params:
                    tools.append({"name": tool_name, "params": params})

        return tools

    def execute_tool(self, tool_name: str, params: Dict) -> Tuple[str, Image.Image]:
        """도구 실행 (비전 도구 포함)"""
        image = None

        if tool_name == "read_file":
            result = self.tools.read_file(
                params.get("file_path", ""),
                int(params.get("offset", 0)),
                int(params.get("limit", 2000))
            )
        elif tool_name == "write_file":
            result = self.tools.write_file(
                params.get("file_path", ""),
                params.get("content", "")
            )
        elif tool_name == "edit_file":
            result = self.tools.edit_file(
                params.get("file_path", ""),
                params.get("old_string", ""),
                params.get("new_string", "")
            )
        elif tool_name == "bash":
            result = self.tools.bash(params.get("command", ""))
        elif tool_name == "glob_files":
            result = self.tools.glob_files(
                params.get("pattern", "*"),
                params.get("path", ".")
            )
        elif tool_name == "grep":
            result = self.tools.grep(
                params.get("pattern", ""),
                params.get("path", "."),
                params.get("file_type")
            )
        elif tool_name == "list_dir":
            result = self.tools.list_dir(params.get("path", "."))
        elif tool_name == "screenshot":
            # 스크린샷 도구 (비전)
            filepath, image = VisionTools.take_screenshot(params.get("region", "full"))
            if image:
                self.current_image = image
                result = f"Screenshot captured: {filepath}\n[Analyzing with vision...]"
            else:
                result = filepath  # Error message
        elif tool_name == "analyze_image":
            # 이미지 분석 도구 (비전)
            filepath, image = VisionTools.load_image(params.get("file_path", ""))
            if image:
                self.current_image = image
                result = f"Image loaded: {filepath}\n[Analyzing with vision...]"
            else:
                result = filepath  # Error message
        else:
            result = f"Unknown tool: {tool_name}"

        return result, image

    def chat(self, user_input: str, image: Image.Image = None) -> str:
        """사용자 입력 처리"""
        self.add_message("user", user_input, image=image)

        # Generate response
        prompt = self.build_prompt()

        # 현재 이미지가 있으면 비전 모드로
        current_img = image or self.current_image

        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            response = self.llm.generate(prompt, image=current_img)

        # Parse and execute tools
        tool_calls = self.parse_tool_calls(response)

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                params = tool_call["params"]

                console.print(f"\n[yellow]Using tool: {tool_name}[/yellow]")

                result, tool_image = self.execute_tool(tool_name, params)

                # Show result
                if len(result) > 500:
                    console.print(Panel(result[:500] + "...", title=f"Tool: {tool_name}"))
                else:
                    console.print(Panel(result, title=f"Tool: {tool_name}"))

                self.add_message("tool", "", tool_name=tool_name, tool_result=result, image=tool_image)

                # 비전 도구 후 이미지 분석
                if tool_image is not None:
                    with console.status("[bold magenta]Analyzing image...[/bold magenta]"):
                        analysis = self.llm.generate(
                            "Describe what you see in this image in detail. "
                            "If it's a screenshot, identify UI elements, text, and layout.",
                            image=tool_image
                        )
                    console.print(Panel(analysis, title="Vision Analysis", border_style="magenta"))

        # Clean response (remove tool tags for display)
        clean_response = re.sub(r'<tool.*?</tool>', '', response, flags=re.DOTALL).strip()

        self.add_message("assistant", response)

        # Clear current image after use
        self.current_image = None

        return clean_response


# ============================================================
# TUI Interface
# ============================================================

def print_banner():
    """배너 출력"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗     ██╗ ██████╗ ██╗   ██╗██╗██████╗  ██████╗ ██████╗   ║
║   ██║     ██║██╔═══██╗██║   ██║██║██╔══██╗██╔════╝██╔═══██╗  ║
║   ██║     ██║██║   ██║██║   ██║██║██║  ██║██║     ██║   ██║  ║
║   ██║     ██║██║▄▄ ██║██║   ██║██║██║  ██║██║     ██║   ██║  ║
║   ███████╗██║╚██████╔╝╚██████╔╝██║██████╔╝╚██████╗╚██████╔╝  ║
║   ╚══════╝╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═════╝  ╚═════╝ ╚═════╝   ║
║                                                               ║
║      LFM2-VL 비전 AI 코딩 어시스턴트 (Claude Code + Vision)     ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")
    console.print(f"[dim]Working directory: {CONFIG.working_dir}[/dim]")
    console.print(f"[dim]Model: {CONFIG.model_name} | Device: {CONFIG.device}[/dim]")
    console.print("[magenta]Vision enabled: screenshot, analyze_image[/magenta]")
    console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]\n")


def print_help():
    """도움말 출력"""
    help_text = """
[bold]Commands:[/bold]
  /help          - Show this help
  /exit, /quit   - Exit LiquidCode
  /clear         - Clear conversation history
  /cd <path>     - Change working directory
  /ls [path]     - List directory
  /model <name>  - Change model
  /status        - Show status
  /screenshot    - Take and analyze screenshot
  /image <path>  - Analyze an image file

[bold]Vision Features (Claude Code에 없음):[/bold]
  - /screenshot: 현재 화면 캡처 후 분석
  - /image <path>: 이미지 파일 분석
  - "스크린샷 찍어줘": 자연어로 스크린샷 요청
  - "이 이미지 분석해줘": 이미지 분석 요청

[bold]Usage:[/bold]
  Just type your request in natural language.
  LiquidCode will read files, edit code, run commands, and SEE images.

[bold]Examples:[/bold]
  "Read the main.py file"
  "Find all Python files with 'test' in the name"
  "Fix the bug in line 42 of utils.py"
  "Take a screenshot and tell me what's on screen"
  "Analyze this image: /path/to/image.png"
"""
    console.print(Panel(help_text, title="LiquidCode Help", border_style="cyan"))


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LiquidCode - AI Coding Assistant with Vision")
    parser.add_argument("--model", default=CONFIG.model_name, help="Model to use")
    parser.add_argument("--dir", default=os.getcwd(), help="Working directory")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    args = parser.parse_args()

    CONFIG.model_name = args.model
    CONFIG.working_dir = args.dir

    print_banner()

    agent = Agent()

    # Initial prompt if provided
    if args.prompt:
        initial_prompt = " ".join(args.prompt)
        console.print(f"[bold green]> {initial_prompt}[/bold green]\n")
        response = agent.chat(initial_prompt)
        console.print(Markdown(response))
        console.print()

    # Interactive loop
    while True:
        try:
            user_input = Prompt.ask("[bold green]>[/bold green]")

            if not user_input.strip():
                continue

            # Commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ["/exit", "/quit"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd == "/help":
                    print_help()
                    continue
                elif cmd == "/clear":
                    agent.messages.clear()
                    console.print("[green]Conversation cleared[/green]")
                    continue
                elif cmd == "/cd":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_dir = os.path.expanduser(parts[1])
                        if os.path.isdir(new_dir):
                            CONFIG.working_dir = os.path.abspath(new_dir)
                            console.print(f"[green]Changed to {CONFIG.working_dir}[/green]")
                        else:
                            console.print(f"[red]Directory not found: {new_dir}[/red]")
                    continue
                elif cmd == "/ls":
                    parts = user_input.split(maxsplit=1)
                    path = parts[1] if len(parts) > 1 else "."
                    result = Tools.list_dir(path)
                    console.print(result)
                    continue
                elif cmd == "/screenshot":
                    console.print("[yellow]Taking screenshot...[/yellow]")
                    filepath, img = VisionTools.take_screenshot()
                    if img:
                        console.print(f"[green]Screenshot saved: {filepath}[/green]")
                        with console.status("[bold magenta]Analyzing...[/bold magenta]"):
                            analysis = agent.llm.generate(
                                "Describe what you see in this screenshot in detail.",
                                image=img
                            )
                        console.print(Panel(analysis, title="Vision Analysis", border_style="magenta"))
                    else:
                        console.print(f"[red]{filepath}[/red]")
                    continue
                elif cmd == "/image":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        filepath, img = VisionTools.load_image(parts[1])
                        if img:
                            console.print(f"[green]Image loaded: {filepath}[/green]")
                            with console.status("[bold magenta]Analyzing...[/bold magenta]"):
                                analysis = agent.llm.generate(
                                    "Describe what you see in this image in detail.",
                                    image=img
                                )
                            console.print(Panel(analysis, title="Vision Analysis", border_style="magenta"))
                        else:
                            console.print(f"[red]{filepath}[/red]")
                    else:
                        console.print("[red]Usage: /image <path>[/red]")
                    continue
                elif cmd == "/status":
                    status = Table(title="Status", box=box.ROUNDED)
                    status.add_column("Item", style="cyan")
                    status.add_column("Value", style="green")
                    status.add_row("Model", CONFIG.model_name)
                    status.add_row("Device", CONFIG.device)
                    status.add_row("Working Dir", CONFIG.working_dir)
                    status.add_row("Messages", str(len(agent.messages)))
                    status.add_row("Vision", "Enabled")
                    if torch.cuda.is_available():
                        mem = torch.cuda.memory_allocated() / 1e9
                        status.add_row("GPU Memory", f"{mem:.2f} GB")
                    console.print(status)
                    continue

            # Chat
            response = agent.chat(user_input)

            if response:
                console.print()
                console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/yellow]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
