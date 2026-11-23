"""
Differences from JAX to note:
- on integer inputs, jax.lax.div does integer division that rounds toward zero, while np.divide always returns floats.
- JAX allows some limited broadcasting in lax
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
from itertools import pairwise

from .core import Primitive


sin_p = Primitive('sin')
sin_p.def_impl(np.sin)

def sin(x):
    return sin_p.bind(x)


cos_p = Primitive('cos')
cos_p.def_impl(np.cos)

def cos(x):
    return cos_p.bind(x)


exp_p = Primitive('exp')
exp_p.def_impl(np.exp)

def exp(x):
    return exp_p.bind(x)


log_p = Primitive('log')
log_p.def_impl(np.log)

def log(x):
    return log_p.bind(x)


neg_p = Primitive('neg')
neg_p.def_impl(np.negative)

def neg(x):
    return neg_p.bind(x)


add_p = Primitive('add')

@add_p.def_impl
def _add_impl(x, y):
    _assert_equal_shapes_dtypes(x, y, add_p)
    return np.add(x, y)

def add(x, y):
    return add_p.bind(x, y)


sub_p = Primitive('sub')

@sub_p.def_impl
def _sub_impl(x, y):
    _assert_equal_shapes_dtypes(x, y, sub_p)
    return np.subtract(x, y)

def sub(x, y):
    return sub_p.bind(x, y)


mul_p = Primitive('mul')

@mul_p.def_impl
def _mul_impl(x, y):
    _assert_equal_shapes_dtypes(x, y, mul_p)
    return np.multiply(x, y)

def mul(x, y):
    return mul_p.bind(x, y)


div_p = Primitive('div')

@div_p.def_impl
def _div_impl(x, y):
    _assert_equal_shapes_dtypes(x, y, div_p)
    return np.divide(x, y)

def div(x, y):
    return div_p.bind(x, y)


greater_p = Primitive('greater')

@greater_p.def_impl
def _greater_impl(x, y):
    _assert_equal_shapes_dtypes(x, y, greater_p)
    return np.greater(x, y)

def greater(x, y):
    return greater_p.bind(x, y)


reduce_sum_p = Primitive('reduce_sum')

@reduce_sum_p.def_impl
def _reduce_sum_impl(x, *, axes: tuple[int, ...]):
    return np.sum(x, axis=axes)

def reduce_sum(x, axes: Iterable[int]):
    return reduce_sum_p.bind(x, axes=tuple(axes))


broadcast_in_dim_p = Primitive('broadcast_in_dim')

@broadcast_in_dim_p.def_impl
def _broadcast_in_dim_impl(x, *, 
                           shape: tuple[int, ...], 
                           broadcast_dimensions: tuple[int, ...]):
    x = np.asarray(x)
    _check_broadcast_dimensions(x, shape, broadcast_dimensions)
    extended_shape = [1 for i in range(len(shape))]
    
    for i in range(x.ndim):
        extended_shape[broadcast_dimensions[i]] = x.shape[i]
    return np.broadcast_to(x.reshape(extended_shape), shape)


def broadcast_in_dim(x, 
                     shape: Iterable[int], 
                     broadcast_dimensions: Iterable[int]):
    """
    broadcast_dimensions says which output axis each of x's axes 
    lands in.
    """
    return broadcast_in_dim_p.bind(x, shape=tuple(shape), 
                                   broadcast_dimensions=tuple(broadcast_dimensions))


transpose_p = Primitive('transpose')

@transpose_p.def_impl
def _transpose_impl(x, *, permutation: tuple[int, ...]):
    return np.transpose(x, axes=permutation)

def transpose(x, permutation: Iterable[int]):
    return transpose_p.bind(x, permutation=tuple(permutation))


convert_element_type_p = Primitive('convert_element_type')

@convert_element_type_p.def_impl
def _convert_element_type_impl(x, *, new_dtype):
    return np.asarray(x).astype(new_dtype)

def convert_element_type(x, new_dtype: npt.DTypeLike):
    return convert_element_type_p.bind(x, new_dtype=np.dtype(new_dtype))


# ------------------------------------- helpers -------------------------------

def _assert_equal_shapes_dtypes(x, y, primitive: Primitive):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.dtype != y.dtype:
        raise TypeError(f'x dtype {x.dtype} and y dtype {y.dtype}'
                        f' disagree in primitive {primitive.name}')

    if x.shape != y.shape:
        raise TypeError(f'x shape {x.shape} and y shape {y.shape}'
                        f' disagree in primitive {primitive.name}')


def _check_broadcast_dimensions(x, shape: tuple[int, ...], 
                                broadcast_dimensions: tuple[int, ...]):

    if x.ndim != len(broadcast_dimensions):
        raise ValueError(f'broadcast_dimensions must have same length as x.ndim'
                         f' for primitive {broadcast_in_dim_p.name}')

    if any(dim < 0 for dim in broadcast_dimensions):
        raise ValueError(f'broadcast_dimensions cannot have negative entries'
                         f' for primitive {broadcast_in_dim_p.name}')

    if any(dim > len(shape) - 1 for dim in broadcast_dimensions):
        raise ValueError(f'broadcast_dimensions[i] must be less than `len(shape)`'
                         f' for primitive {broadcast_in_dim_p.name}')

    if not all(a < b for a, b in pairwise(broadcast_dimensions)):
        raise ValueError(f'broadcast_dimensions tuple must be strictly'
                         f' increasing for primitive {broadcast_in_dim_p.name}')

