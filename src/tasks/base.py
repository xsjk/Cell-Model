from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskContext:
    """Runtime context shared with a task instance.

    Attributes:
        task_name: Unique identifier registered in the task registry.
        config: Parsed configuration mapping (typically loaded from TOML).
        workspace: Root directory of the repository.
        output_dir: Directory where the task can store checkpoints, logs, etc.
    """

    task_name: str
    config: Mapping[str, Any]
    workspace: Path
    output_dir: Path


class BaseTask(ABC):
    """Base class for all task implementations."""

    #: Name used when registering the task. Subclasses must override.
    name: str = "base"

    #: Optional human readable description surfaced via the CLI.
    description: str | None = None

    #: Default relative path (inside ``config/tasks/<task>/``) of the task
    #: configuration file. Tasks can override when they need different layout.
    default_config_filename: str = "task.toml"

    def __init__(self, context: TaskContext) -> None:
        self._context = context

    # ------------------------------------------------------------------
    # Required hooks – concrete tasks must implement these.
    # ------------------------------------------------------------------
    @abstractmethod
    def build_dataloaders(self) -> Any:
        """Create task specific dataloaders (train/val/test).

        The return value is task dependent, but should contain everything the
        trainer needs (e.g. a dataclass, dict or namespace).
        """

    @abstractmethod
    def build_model(self) -> Any:
        """Construct and return the model (or model components)."""

    @abstractmethod
    def build_trainer(self) -> Any:
        """Construct the trainer/optimizer orchestration object."""

    @abstractmethod
    def run_train(self) -> None:
        """Execute the training loop for this task."""

    # ------------------------------------------------------------------
    # Optional hooks – tasks can override when applicable.
    # ------------------------------------------------------------------
    def run_evaluation(self) -> None:
        """Optional evaluation entry point."""
        raise NotImplementedError(f"Task '{self.name}' does not implement evaluation yet")

    def run_inference(self) -> None:
        """Optional inference entry point."""
        raise NotImplementedError(f"Task '{self.name}' does not implement inference yet")

    # ------------------------------------------------------------------
    # Convenience accessors.
    # ------------------------------------------------------------------
    @property
    def context(self) -> TaskContext:
        """Expose the immutable runtime context."""
        return self._context

    @property
    def config(self) -> Mapping[str, Any]:
        """Shortcut for ``context.config``."""
        return self._context.config

    @property
    def output_dir(self) -> Path:
        """Directory where the task can persist artifacts."""
        return self._context.output_dir
