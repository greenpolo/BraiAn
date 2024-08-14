import copy

__all__ = ["deflect"]

def keep_type(F, obj, attr: str): # a function decorator
    # if the result of the wrapped function is of the same type as attr,
    # it creates a copy of obj and set the attr to the new
    attr_obj = obj.__dict__[attr]
    def wrapper(*args, inplace=False, **kwargs): # on wrapped function call
        # if some of the arguments have the same type as the caller, deflect to use arg.<attr>
        args = [arg if type(arg) != type(obj) else arg.__dict__[attr] for arg in args]
        kwargs = {kw: arg if type(arg) != type(obj) else arg.__dict__[attr] for kw,arg in kwargs.items()}
        result = F(*args, **kwargs)
        if type(result) == type(attr_obj):
            if inplace:
                obj.__setattr__(attr, result)
                return obj
            else:
                _obj = copy.copy(obj)
                _obj.__setattr__(attr, result)
                return _obj
        return result
    return wrapper

def find_in_object(obj, attr):
    for clazz in type(obj).__mro__:
        if attr not in clazz.__dict__:
            continue
        return clazz.__dict__[attr].__get__(obj)
    raise TypeError(f"unsupported operand type(s) for {attr}")

def deflect_call(target: str, op: str):
    # most of arithmetic operation are redirected __getattr__
    def op_wrapper(obj, *args, **kwargs):
        if target not in obj.__dict__:
            raise AttributeError(f"'{type(obj)}' object has no attribute '{target}'")
        target_obj = obj.__dict__[target]
        if op == "__getattr__":
            attr = args[0]
            # __getattr__ is called with one argument and expects to return a property or a function
            deflected_fun = find_in_object(target_obj, attr)
            return keep_type(deflected_fun, obj, target)
        else:
            # other functions expect a result!
            deflected_fun = find_in_object(target_obj, op)
            return keep_type(deflected_fun, obj, target)(*args, **kwargs)

    return op_wrapper

def deflect(on_attribute: str,
            arithmetics=True,
            container=True):
    """
    Constructor for deflecting metaclasses. A class using the resulting metaclass will
    deflect all access to unknown attributes to a _target_ attribute (i.e. `on_attribute`).
    If the result of the deflected call has the same type as the target,
    it creates a copy of the instance and with `on_attribute` substituted.\\
    All deflected functions expose `inplace` parameter (defaults to `False`) that,
    if `True`, changes the target attribute instance with the result of the computation.

    Parameters
    ----------
    on_attribute
        The name of the attribute where to deflect all unknown attribute access.
    arithmetics
        Whether to deflect [arithmetic operations](https://docs.python.org/3.3/reference/datamodel.html#emulating-numeric-types)
        or not.
    container
        Whether to deflect [container operations](https://docs.python.org/3.3/reference/datamodel.html#emulating-container-types)
        or not.

    Returns
    -------
    :
        A metaclass.
    """
    class Deflector(type):
        def __new__(meta, classname, supers, classdict):
            classdict["__getattr__"] = deflect_call(on_attribute, "__getattr__")
            if arithmetics:
                # from https://docs.python.org/3.3/reference/datamodel.html#emulating-numeric-types
                for op in ["__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
                           "__mod__", "__divmod__", "__pow__", "__lshift__", "__rshift__",
                           "__and__", "__xor__", "__or__", "__neg__", "__pos__", "__abs__",
                           "__invert__"]:
                    classdict[op] = deflect_call(on_attribute, op)
            if container:
                # from https://docs.python.org/3.3/reference/datamodel.html#emulating-container-types
                for op in ["__len__", "__getitem__", "__setitem__", "__delitem__", "__iter__", "__reversed__", "__contains__"]:
                    classdict[op] = deflect_call(on_attribute, op)
            return type.__new__(meta, classname, supers, classdict)
    return Deflector

if __name__ == "__main__":
    class C(metaclass=deflect(on_attribute="num", arithmetics=True)):
        def __init__(self, n):
            self.num = n
        def __repr__(self) -> str:
            return f"C(num={type(self.num)})"

    class A_():
        def ciao(self):
            print("A_")
    class A(A_):
        pass
    class B(object):
        def ciao(self):
            print("B")
    class D(A, B):
        pass

    import numpy as np
    c = C(np.array([1,2,3]))
    print(f"c.num: {c.num}")
    # >>> c.num: [1 2 3]
    a = c.__add__(2.1)
    print(a, type(a))
    # >>> C(num=<class 'numpy.ndarray'>) <class '__main__.C'>
    a = c.sum()
    print(a, type(a))
    # >>> 6 <class 'numpy.int64'>
    a = (c + 2.1)
    print(a, type(a), a.num)
    # >>> C(num=<class 'numpy.ndarray'>) <class '__main__.C'> [3.1 4.1 5.1]
    c_ = C(np.array([1,2,3]))
    print(f"c_ is (c_ + 1):", c_ is (c + 1))
    # >>> c_ is (c_ + 1): False
    print(f"c_ is (c_.__add__(1, inplace=True)):", c_ is (c_.__add__(1, inplace=True)))
    # >>> c_ is (c_.__add__(1, inplace=True)): True
    print(f"c_ is (c_.clip(max=3)):", c_ is (c_.clip(max=3)))
    # >>> c_ is (c_.clip(max=3)): False
    print(f"c_ is (c_.clip(max=3, inplace=True)):", c_ is (c_.clip(max=3, inplace=True)))
    # >>> c_ is (c_.clip(max=3, inplace=True)): True
    print("c_.num:", c_.num)
    # >>> c_.num: [2 3 3]
    print("c / c_:", r:=(c / c_), r.num)
    # >>> c / c_: C(num=<class 'numpy.ndarray'>) [0.5        0.66666667 1.        ]
    D().ciao()
    # >>> A_
    C(D()).ciao()
    # >>> A_