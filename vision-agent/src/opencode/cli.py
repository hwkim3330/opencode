"""OpenCode CLI - Vision Agent Interface"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="opencode",
    help="Vision-based AI Agent - See and automate your screen",
    add_completion=False,
)
console = Console()


def get_agent():
    """Lazy load agent"""
    from .agent.core import VisionAgent
    return VisionAgent()


@app.command()
def describe(
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save screenshot to file"),
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Describe what's currently on screen"""
    console.print("[bold blue]Capturing screen...[/]")

    agent = get_agent()
    agent.load()

    screen = agent.see(monitor)
    if save:
        screen.save(save)
        console.print(f"[green]Screenshot saved to {save}[/]")

    console.print("[bold blue]Analyzing...[/]")
    result = agent.describe(screen)

    console.print(Panel(result, title="Screen Description", border_style="green"))


@app.command()
def find(
    element: str = typer.Argument(..., help="UI element to find"),
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Find a specific UI element on screen"""
    console.print(f"[bold blue]Looking for '{element}'...[/]")

    agent = get_agent()
    agent.load()

    result = agent.find(element)
    console.print(Panel(result, title=f"Finding: {element}", border_style="green"))


@app.command()
def error(
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Read and explain any error message on screen"""
    console.print("[bold blue]Looking for errors...[/]")

    agent = get_agent()
    agent.load()

    result = agent.read_error()
    console.print(Panel(result, title="Error Analysis", border_style="red"))


@app.command()
def ocr(
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save text to file"),
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Extract all text from screen (OCR)"""
    console.print("[bold blue]Extracting text...[/]")

    agent = get_agent()
    agent.load()

    result = agent.ocr()

    if save:
        Path(save).write_text(result)
        console.print(f"[green]Text saved to {save}[/]")

    console.print(Panel(result, title="Extracted Text", border_style="green"))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the screen"),
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Ask any question about what's on screen"""
    console.print(f"[bold blue]Analyzing: {question}[/]")

    agent = get_agent()
    agent.load()

    result = agent.ask(question)
    console.print(Panel(result, title="Answer", border_style="green"))


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    max_steps: int = typer.Option(10, "--steps", "-n", help="Maximum steps"),
):
    """Execute a task by looking at screen and taking actions (experimental)"""
    console.print(f"[bold yellow]Executing task: {task}[/]")
    console.print("[dim]Press Ctrl+C to abort. Move mouse to corner for emergency stop.[/]")

    agent = get_agent()
    agent.load()

    results = agent.execute_task(task, max_steps)

    for r in results:
        console.print(f"Step {r['step']}: {r['action']}")

    console.print("[bold green]Task complete![/]")


@app.command()
def screenshot(
    output: str = typer.Argument("screenshot.png", help="Output filename"),
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor number"),
):
    """Take a screenshot"""
    from .vision.capture import ScreenCapture

    capture = ScreenCapture()
    screen = capture.capture_full(monitor)
    screen.save(output)
    console.print(f"[green]Screenshot saved to {output}[/]")


@app.command()
def monitors():
    """List available monitors"""
    from .vision.capture import ScreenCapture

    capture = ScreenCapture()
    mons = capture.list_monitors()

    console.print("[bold]Available Monitors:[/]")
    for m in mons:
        console.print(f"  {m['index']}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")


@app.command()
def interactive():
    """Interactive mode - chat with the agent about your screen"""
    console.print(Panel.fit(
        "[bold green]OpenCode Vision Agent[/]\n"
        "Type questions about your screen. Commands:\n"
        "  /screenshot - take screenshot\n"
        "  /describe - describe screen\n"
        "  /find <element> - find UI element\n"
        "  /ocr - extract text\n"
        "  /quit - exit",
        title="Interactive Mode"
    ))

    agent = get_agent()
    console.print("[dim]Loading model...[/]")
    agent.load()
    console.print("[green]Ready![/]")

    while True:
        try:
            user_input = console.input("[bold cyan]> [/]").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                break
            elif user_input == "/screenshot":
                screen = agent.see()
                screen.save("screenshot.png")
                console.print("[green]Saved to screenshot.png[/]")
            elif user_input == "/describe":
                result = agent.describe()
                console.print(Panel(result, border_style="green"))
            elif user_input.startswith("/find "):
                element = user_input[6:]
                result = agent.find(element)
                console.print(Panel(result, border_style="green"))
            elif user_input == "/ocr":
                result = agent.ocr()
                console.print(Panel(result, border_style="green"))
            else:
                result = agent.ask(user_input)
                console.print(Panel(result, border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

    console.print("[dim]Goodbye![/]")


def main():
    app()


if __name__ == "__main__":
    main()
