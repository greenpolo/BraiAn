import copy

def keep_type(F, obj, attr): # a function decorator        
    # if the result of the wrapped function is of the same type as attr,
    # it creates a copy of obj and set the attr to the new result
    attr_obj = obj.__dict__[attr]
    def wrapper(*args, **kwargs): # on wrapped function call
        result = F(*args, **kwargs)
        print("result:", type(result))
        print(f"attr ('{attr}'):", type(attr_obj))
        if type(result) == type(attr_obj):
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
                for op in ["__add__", "__sub__", "__mul__", "__div__", "__invert__", "__neg__", "__pos__"]:
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