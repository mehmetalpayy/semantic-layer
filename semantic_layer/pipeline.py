"""Pipeline base classes and components used in semantic layer tests."""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hamilton.async_driver import AsyncDriver
from hamilton.driver import Driver
from haystack import Pipeline


class BasicPipeline(metaclass=ABCMeta):
    """Abstract wrapper for synchronous or async pipelines."""

    def __init__(self, pipe: Pipeline | AsyncDriver | Driver):
        """Store the underlying pipeline instance."""
        self._pipe = pipe

    @abstractmethod
    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Execute the pipeline and return its outputs."""
        ...


@dataclass
class PipelineComponent(Mapping):
    """Mapping container for pipeline dependencies."""

    llm_provider: Any = None
    embedder_provider: Any = None
    document_store_provider: Any = None

    def __getitem__(self, key):
        """Return component by attribute name."""
        return getattr(self, key)

    def __iter__(self):
        """Iterate over component keys."""
        return iter(self.__dict__)

    def __len__(self):
        """Return the number of components."""
        return len(self.__dict__)
