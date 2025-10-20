from dataclasses import dataclass

from .base import BaseTask, TaskContext


@dataclass(frozen=True)
class TaskInfo:
    name: str
    cls: type[BaseTask]
    description: str | None = None


class TaskRegistry:
    _registry: dict[str, TaskInfo] = {}

    @classmethod
    def register(cls, name: str, task_cls: type[BaseTask], description: str | None = None) -> None:
        key = name.lower()
        if key in cls._registry:
            existing = cls._registry[key]
            raise ValueError(f"Task '{name}' already registered with class {existing.cls.__name__}")
        cls._registry[key] = TaskInfo(name=name, cls=task_cls, description=description)

    @classmethod
    def create(cls, name: str, context: TaskContext) -> BaseTask:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            raise KeyError(f"Unknown task '{name}'. Registered tasks: {available}")
        info = cls._registry[key]
        return info.cls(context)

    @classmethod
    def info(cls, name: str) -> TaskInfo | None:
        return cls._registry.get(name.lower())

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._registry)

    @classmethod
    def all(cls) -> tuple[TaskInfo, ...]:
        return tuple(cls._registry.values())


def register_task(task_cls: type[BaseTask]) -> type[BaseTask]:
    """Decorator that registers a ``BaseTask`` subclass.

    The decorated class must define the ``name`` class attribute. The optional
    ``description`` attribute is picked up when present.
    """

    if not issubclass(task_cls, BaseTask):
        raise TypeError("Only BaseTask subclasses can be registered")

    name = getattr(task_cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Registered tasks must define a non-empty 'name' class attribute")

    description = getattr(task_cls, "description", None)
    if description is not None and not isinstance(description, str):
        raise ValueError("Task 'description' must be a string when provided")

    TaskRegistry.register(name, task_cls, description)
    return task_cls
