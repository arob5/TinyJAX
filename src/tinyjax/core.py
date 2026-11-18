from __future__ import annotations
from typing import Any, Callable
from contextlib import contextmanager

class Primitive:
    """A primitive operation

    Rule tables are keyed by primitive objects. Two primitives with the same 
    name are different primitives; comparison is by identity. 
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def bind(self, *args, **params):
        """ Single entry point for interpreting an operation

        The operation is interpreted by the current trace.
        """
        return get_current_trace().process_primitive(self, args, params)


impl_rules: dict[Primitive, Callable] = {}


class Trace:
    """ Interpreter for primitive operations

    We call this a "trace" to match JAX, but "interpreter" would be a better
    name here. Note that one trace handles all primitive operations. 
    """
    def process_primitive(self, 
                          primitive: Primitive, 
                          args: tuple[Any, ...], 
                          params: dict[str, Any]):
        raise NotImplementedError("subclass must implement this")


class EvalTrace(Trace):
    """ Evaluation interpreter

    Evaluates primitives on numpy arrays at the bottom of the trace chain.
    """

    def process_primitive(self, 
                          primitive: Primitive, 
                          args: tuple[Any, ...], 
                          params: dict[str, Any]):
        rule = impl_rules.get(primitive)
        if rule is None:
            raise NotImplementedError(f"no impl rule for {primitive}")
        else: 
            return rule(*args, **params)


current_trace: Trace = EvalTrace()

def get_current_trace() -> Trace:
    return current_trace

@contextmanager
def set_current_trace(trace: Trace):
    """
    Sets current trace to `trace`. Restores trace to previous trace
    upon exiting the with block.
    """
    global current_trace
    previous_trace = get_current_trace()
    try:
        current_trace = trace
        yield
    finally:
        current_trace = previous_trace