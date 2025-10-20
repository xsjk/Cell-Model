from pathlib import Path

import typer

from .commands import compress, fetch, inspect

app = typer.Typer(help="Data processing tasks for Cell-Model.")


def _config_arg_to_str(config: Path | None) -> str | None:
    return str(config) if config is not None else None


default_config = typer.Option(
    None,
    "--config",
    "-c",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to dataset configuration TOML (defaults to config/dataset.toml).",
)


@app.command("fetch")
def fetch_main(config: Path | None = default_config) -> None:
    fetch.main(_config_arg_to_str(config))


@app.command("compress")
def compress_main(config: Path | None = default_config) -> None:
    compress.main(_config_arg_to_str(config))


@app.command("inspect")
def inspect_main(config: Path | None = default_config) -> None:
    inspect.main(_config_arg_to_str(config))


@app.command("all")
def run_all(config: Path | None = default_config) -> None:
    config_path = _config_arg_to_str(config)
    fetch.main(config_path)
    compress.main(config_path)
    inspect.main(config_path)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
