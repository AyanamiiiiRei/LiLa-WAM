"""A tiny name -> class registry for dataset features.

Adding an input modality (e.g. a history image for control experiment B) is:
    1. write a class in data/features.py (or a new module) decorated with
       ``@FEATURES.register("my_feature")``
    2. list "my_feature" in ``dataset.features`` of an experiment config
No existing code has to change.
"""

from __future__ import annotations

from typing import Callable, Dict, Type


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def deco(cls: Type) -> Type:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} registered twice")
            self._items[name] = cls
            cls.registry_name = name
            return cls
        return deco

    def get(self, name: str) -> Type:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


FEATURES = Registry("dataset feature")
