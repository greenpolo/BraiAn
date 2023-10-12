import copy

def keep_type(F, obj, attr: str): # a function decorator        
    # if the result of the wrapped function is of the same type as attr,
    # it creates a copy of obj and set the attr to the new result
    attr_obj = obj.__dict__[attr]
    def wrapper(*args, inplace=False, **kwargs): # on wrapped function call
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

def deflect_call(target: str, op: str):
    # most of arithmetic operation are redirected __getattr__ 
    def op_wrapper(obj, *args, **kwargs):
        if target not in obj.__dict__:
            raise AttributeError(f"'{type(obj)}' object has no attribute '{target}'")
        target_obj = obj.__dict__[target]
        if op == "__getattr__":
            # __getattr__ is called with one argument and expects to return a property or a function
            attr = args[0]
            deflected_fun = type(target_obj).__dict__[attr].__get__(target_obj)
            return keep_type(deflected_fun, obj, target)
        elif op in type(target_obj).__dict__:
            # other functions expect a result!
            deflected_fun = type(target_obj).__dict__[op].__get__(target_obj)
            return keep_type(deflected_fun, obj, target)(*args, **kwargs)
        else:
            raise TypeError(f"unsupported operand type(s) for {op}")
    return op_wrapper

def deflect(on_attribute: str, arithmetics=True):
    class Deflector(type):
        def __new__(meta, classname, supers, classdict):
            classdict["__getattr__"] = deflect_call(on_attribute, "__getattr__")
            if arithmetics:
                for op in ["__add__", "__sub__", "__mul__", "__floordiv__", "__truediv__", "__invert__", "__neg__", "__pos__"]:
                    classdict[op] = deflect_call(on_attribute, op)
            return type.__new__(meta, classname, supers, classdict)
    return Deflector

class C(metaclass=deflect(on_attribute="num", arithmetics=True)):
    def __init__(self, n):
        self.num = n
    def __repr__(self) -> str:
        return f"C(num={type(self.num)})"

if __name__ == "__main__":
    import numpy as np
    c = C(np.array([1,2,3]))
    print(f"c.num: {c.num}")
    a = c.__add__(2.1)
    print(a, type(a))
    a = c.sum()
    print(a, type(a))
    a = (c + 2.1)
    print(a, type(a), a.num)
    c_ = C(np.array([1,2,3]))
    print(f"c_ is (c_ + 1):", c is (c + 1))
    print(f"c_ is (c_.__add__(1, inplace=True)):", c_ is (c_.__add__(1, inplace=True)))
    print(f"c_ is (c_.clip(max=3)):", c_ is (c_.clip(max=3)))
    print(f"c_ is (c_.clip(max=3, inplace=True)):", c_ is (c_.clip(max=3, inplace=True)))
    print("c_.num:", c_.num)
    print("c / c_:", r:=(c / c_), r.num)