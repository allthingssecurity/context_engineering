"""Stable operator interface, pluggable registry, and the pipeline runner.

Every stage in the production pipeline is an :class:`Operator` with a stable
``name``/``version`` and a ``run(state) -> state`` method.  The base ``run``
wrapper times the operator, records an :class:`OperatorMetrics`, and — crucially
— **never lets an operator silently fail**: an exception is captured into the
state's ``errors``/``warnings``/``trace`` and the pipeline continues (unless the
runner is in strict mode).

Adding a new operator is a two-line affair: subclass :class:`Operator` and
register it.  See ``registry.py`` for the defaults.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple, Union

from .state import ContextBuildState, OperatorMetrics


class Operator(ABC):
    """Base class for every pipeline stage.

    Subclasses set ``name``/``version`` and implement :meth:`_run`.  Config is
    passed as keyword arguments and stored on ``self.config``.
    """

    name: str = "operator"
    version: str = "1.0"

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abstractmethod
    def _run(self, state: ContextBuildState) -> ContextBuildState:
        """Do the work; mutate and return ``state``."""

    def run(self, state: ContextBuildState) -> ContextBuildState:
        """Timed, fail-safe wrapper around :meth:`_run`."""
        start = perf_counter()
        candidates_in = len(state.candidates)
        error = None
        try:
            state = self._run(state)
        except Exception as exc:  # never silently fail
            error = f"{type(exc).__name__}: {exc}"
            state.errors.append(f"{self.name}: {error}")
            state.warnings.append(f"operator '{self.name}' failed: {error}")
            state.log(self.name, self.version, "error", error=error)
        duration_ms = round((perf_counter() - start) * 1000, 3)
        state.metrics.append(
            OperatorMetrics(
                operator=self.name,
                version=self.version,
                duration_ms=duration_ms,
                candidates_in=candidates_in,
                candidates_out=len(state.candidates),
                error=error,
            )
        )
        return state


# A factory is either an Operator subclass or any callable(**config) -> Operator.
OperatorFactory = Callable[..., Operator]
# A pipeline spec entry: "name" or ("name", {config}).
SpecEntry = Union[str, Tuple[str, Dict[str, Any]]]


class OperatorRegistry:
    """Pluggable registry mapping operator names to factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, OperatorFactory] = {}

    def register(self, name: str, factory: OperatorFactory) -> None:
        """Register an operator factory under ``name`` (overwrites)."""
        self._factories[name] = factory

    def create(self, name: str, **config: Any) -> Operator:
        if name not in self._factories:
            raise KeyError(f"unknown operator: {name}")
        return self._factories[name](**config)

    def available(self) -> List[str]:
        return sorted(self._factories)


class Pipeline:
    """An ordered list of operator instances (a concrete ``F_s``)."""

    def __init__(self, operators: List[Operator]) -> None:
        self.operators = operators

    @property
    def signature(self) -> List[str]:
        return [op.name for op in self.operators]

    def run(
        self, state: ContextBuildState, raise_on_error: bool = False
    ) -> ContextBuildState:
        for op in self.operators:
            state = op.run(state)
            if raise_on_error and state.errors:
                raise RuntimeError(state.errors[-1])
        return state


def build_pipeline(registry: OperatorRegistry, spec: List[SpecEntry]) -> Pipeline:
    """Instantiate a :class:`Pipeline` from a registry and a spec list.

    ``spec`` entries are either ``"operator_name"`` or
    ``("operator_name", {config...})``.
    """
    operators: List[Operator] = []
    for entry in spec:
        if isinstance(entry, str):
            operators.append(registry.create(entry))
        else:
            name, config = entry
            operators.append(registry.create(name, **config))
    return Pipeline(operators)
