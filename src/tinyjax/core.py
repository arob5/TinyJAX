"""
JAX works by defining a small set of primitive operations (add, multiply, etc.). 
Of course, we want to be able to evaluate these operations. However, in certain 
contexts we want to interpret them in other ways: as their Jacobian-vector 
product, or as their batched/vectorized form, or to stage them into a compiled 
program (the basis of JIT compilation). JAX calls such an interpreter a 
"trace", and the current trace is managed by a context manager. When a 
primitive operation is requested, the primitive is handed to the current trace
to interpret it (which might mean evaluating it, computing its Jacobian-vector
product, etc.)
"""
from __future__ import annotations
from typing import Any, Callable
from contextlib import contextmanager


class Primitive:
    """A primitive operation

    Defined by its name and implementation, the function used to evaluate it.
    Two primitives with the same name are different primitives; comparison is 
    by identity. 
    """
    def __init__(self, name: str):
        self.name = name
        self.impl: Callable|None = None

    def __repr__(self):
        return self.name

    def def_impl(self, rule: Callable) -> Callable:
        """ Setter for implementation rule

        The rule is the evaluation rule defining the primitive. This can
        be used as a decorator.
        """
        self.impl = rule
        return rule

    def bind(self, *args, **params):
        """ Single entry point for interpreting an operation

        The operation is interpreted by the current trace.
        """
        return get_current_trace().process_primitive(self, args, params)


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

    Difference from JAX: In JAX, the impl for most primitives compiles that 
    operation with XLA. This is why a first call to a primitive in JAX will
    be slower than future calls. We don't implement XLA compilation here.
    """

    def process_primitive(self, 
                          primitive: Primitive, 
                          args: tuple[Any, ...], 
                          params: dict[str, Any]):
        rule = primitive.impl
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