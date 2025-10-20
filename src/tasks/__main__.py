import json
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import PROJECT_ROOT, load_task_config

from . import TaskContext, TaskInfo, TaskRegistry

app = typer.Typer(help="Manage Cell-Model tasks.")
console = Console()


def collect_task_infos() -> list[TaskInfo]:
    return sorted(TaskRegistry.all(), key=lambda info: info.name)


def format_tasks_as_text(task_infos: list[TaskInfo]) -> Table | Text:
    if not task_infos:
        return Text("No registered tasks.", style="bold yellow")

    table = Table(title="Registered Tasks", header_style="bold cyan", expand=True)
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Class", style="dim")

    for info in task_infos:
        qualified_name = f"{info.cls.__module__}.{info.cls.__qualname__}"
        table.add_row(info.name, info.description or "—", qualified_name)

    return table


def format_tasks_as_json(task_infos: list[TaskInfo]) -> str:
    payload = [
        {
            "name": info.name,
            "description": info.description,
            "class": f"{info.cls.__module__}.{info.cls.__qualname__}",
        }
        for info in task_infos
    ]
    return json.dumps(payload, indent=2)


def echo_task_listing(json_output: bool) -> None:
    task_infos = collect_task_infos()
    if json_output:
        typer.echo(format_tasks_as_json(task_infos))
    else:
        console.print(format_tasks_as_text(task_infos))


def _build_task_choice() -> click.Choice:
    names = [info.name for info in collect_task_infos()]
    return click.Choice(names, case_sensitive=False)


TASK_NAME_CHOICE = _build_task_choice()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the task list as JSON (when no subcommand is provided)."),
) -> None:
    if ctx.invoked_subcommand is None:
        echo_task_listing(json_output)


@app.command("list")
def list_tasks(json_output: bool = typer.Option(False, "--json", help="Emit the task list as JSON.")) -> None:
    echo_task_listing(json_output)


@app.command("train")
def train(
    task: str = typer.Argument(..., metavar="TASK", help="Registered task name to train.", click_type=TASK_NAME_CHOICE),
    task_config: Path | None = typer.Option(None, "--task-config", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Path to a TOML config file overriding the default."),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", help="Directory for experiment artifacts."),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace root directory (defaults to repository root)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Initialise the task without training."),
) -> None:
    config = load_task_config(task, str(task_config) if task_config else None)

    workspace_dir = workspace.expanduser().resolve() if workspace else PROJECT_ROOT
    output_dir = output_dir.expanduser().resolve() / task
    output_dir.mkdir(parents=True, exist_ok=True)

    context = TaskContext(task_name=task, config=config, workspace=workspace_dir, output_dir=output_dir)
    task_instance = TaskRegistry.create(task, context)

    if dry_run:
        typer.echo(f"Loaded task '{task}' (config source: {task_config or 'default'})")
        return

    task_instance.run_train()


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
