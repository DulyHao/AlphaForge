from collections import namedtuple

import numpy as np
from alphagen.data.expression import *

# OPERATORS = [
#     # Unary
#     # Abs,  
#     # Sign,
#     # Log,
#     Inv,
#     S_log1p,
#     # CSRank,

#     # Binary,
#     Add, Sub, Mul, Div, 
#     Pow,
#     # Greater, Less,

#     # Rolling
#     Ref, ts_mean, ts_sum, ts_std, ts_var,  
#     # ts_skew, 
#     # ts_kurt,
#     ts_max, ts_min,
#     ts_med, ts_mad,  
#     # ts_rank,

#     ts_div,
#     ts_pctchange,
#     # ts_ir,
#     # ts_min_max_diff,
#     # ts_max_diff,ts_min_diff,
#     ts_delta, ts_wma, ts_ema,

#     # Pair rolling
#     ts_cov, ts_corr
# ]

GenericOperator = namedtuple('GenericOperator', ['name', 'function', 'arity'])

unary_ops = [Inv, S_log1p]
binary_ops = [Add, Sub, Mul, Div, Pow,]
rolling_ops = [Ref, ts_mean, ts_sum, ts_std, ts_var, ts_max, ts_min,ts_med, ts_mad,ts_div, ts_pctchange, ts_delta, ts_wma, ts_ema,]
rolling_binary_ops = [ts_cov, ts_corr]

def unary(cls):
    def _calc(a):
        n = len(a)
        return np.array([f'{cls.__name__}({a[i]})' for i in range(n)])

    return _calc


def binary(cls):
    def _calc(a, b):
        n = len(a)
        a = a.astype(str)
        b = b.astype(str)
        return np.array([f'{cls.__name__}({a[i]},{b[i]})' for i in range(n)])

    return _calc

def rolling(cls, day):
    def _calc(a):
        n = len(a)
        return np.array([f'{cls.__name__}({a[i]},{day})' for i in range(n)])

    return _calc

def rolling_binary(cls, day):
    def _calc(a, b):
        n = len(a)
        a = a.astype(str)
        b = b.astype(str)
        return np.array([f'{cls.__name__}({a[i]},{b[i]},{day})' for i in range(n)])

    return _calc

funcs: List[GenericOperator] = []
for op in unary_ops:
    funcs.append(GenericOperator(function=unary(op), name=op.__name__, arity=1))
for op in binary_ops:
    funcs.append(GenericOperator(function=binary(op), name=op.__name__, arity=2))
for op in rolling_ops:
    for day in [10, 20, 30, 40, 50]:
        funcs.append(GenericOperator(function=rolling(op, day), name=op.__name__ + str(day), arity=1))
for op in rolling_binary_ops:
    for day in [10, 20, 30, 40, 50]:
        funcs.append(GenericOperator(function=rolling_binary(op, day), name=op.__name__ + str(day), arity=2))
