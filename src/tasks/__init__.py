import pkgutil

from .base import BaseTask, TaskContext
from .registry import TaskInfo, TaskRegistry, register_task

__all__ = [
    "BaseTask",
    "TaskContext",
    "TaskRegistry",
    "TaskInfo",
    "register_task",
    "iter_task_package_names",
]


def iter_task_package_names() -> list[str]:
    return [f"{__name__}.{name}" for _, name, ispkg in pkgutil.iter_modules(__path__) if ispkg]
