from __future__ import annotations
import numpy as np

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


def _assert_equal_shapes_dtypes(x, y, primitive):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.dtype != y.dtype:
        raise TypeError(f'x dtype {x.dtype} and y dtype {y.dtype}'
                        f' disagree in primitive {primitive.name}')

    if x.shape != y.shape:
        raise TypeError(f'x shape {x.shape} and y shape {y.shape}'
                        f' disagree in primitive {primitive.name}')
    
