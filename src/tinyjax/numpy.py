from __future__ import annotations
from collections.abc import Iterable

from . import lax


def _broadcast_shapes(shape_a: Iterable[int], 
                      shape_b: Iterable[int]) -> tuple:
    shape_a = tuple(shape_a)
    shape_b = tuple(shape_b)

    if len(shape_a) < len(shape_b):
        shorter = shape_a
        longer = shape_b
    else:
        shorter = shape_b
        longer = shape_a

    n = len(longer) - len(shorter)
    shorter = (1,) * n + shorter

    result = ()
    for i, j in zip(shorter, longer):
        if i == j:
            result += (i,)
        elif i == 1:
            result += (j,)
        elif j == 1:
            result += (i,)
        else:
            raise ValueError(f'Shapes {shape_a} and {shape_b} cannot be broadcast.')

    return result


def _broadcast_to(x, shape: Iterable[int]):
    """
    The NumPy convention for broadcasting: line x up with the rightmost axes
    of shape. This is NumPy's "pad 1s on the left" rule.
    """
    shape = tuple(shape)
    if x.shape == shape:
        return x
    
    broadcast_dims = tuple(range(len(shape) - x.ndim, len(shape)))
    return lax.broadcast_in_dim(x, shape=shape, broadcast_dimensions=broadcast_dims)


def add(x, y):
    shape = _broadcast_shapes(x.shape, y.shape)
    x = _broadcast_to(x, shape)
    y = _broadcast_to(y, shape)
    return lax.add(x, y)

