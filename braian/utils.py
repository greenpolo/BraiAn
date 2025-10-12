import errno
import functools
import igraph as ig
import inspect
import itertools
import os
import numpy as np
import pandas as pd
import platform
import requests
import warnings

from collections.abc import Callable, Collection, Sequence
from importlib import resources
from pathlib import Path, WindowsPath

__all__ = ["cache", "classproperty", "deprecated", "get_resource_path", "resource"]

match platform.system():
    case "Windows":
        import win32com.client # type: ignore
        shell = win32com.client.Dispatch("WScript.Shell")
        def resolve_symlink(path: str|WindowsPath):
            if isinstance(path, WindowsPath):
                return shell.CreateShortCut(path).Targetpath if path.suffix == ".lnk" else path
            return shell.CreateShortCut(path).Targetpath if path.endswith(".lnk") else path
    case _:
        def resolve_symlink(path: Path):
            return path.resolve(strict=True)

def cache(filepath: Path|str, url):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if filepath.exists():
        return
    resp = requests.get(url)
    filepath.resolve(strict=False).parent\
            .mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(resp.content)

def merge_ordered(*xs: Sequence, raises: bool=True) -> Sequence:
    vs_obj = list(functools.reduce(set.union, [set(x) for x in xs]))
    n_vertices = len(vs_obj)
    vs = list(range(n_vertices))
    vs_obj2id = dict(zip(vs_obj, vs))
    es = itertools.chain(*(zip(x, x[1:]) for x in xs))
    es = [(vs_obj2id[v1], vs_obj2id[v2]) for v1,v2 in es]
    g = ig.Graph(n_vertices, es, directed=True)
    g.vs["obj"] = vs_obj
    if g.is_dag():
        # topological sorting is possible only if g is a DAG
        return g.vs[g.topological_sorting(mode="out")]["obj"]
    elif raises:
        raise ValueError("The given sequences are not sorted in a compatible way.")
    else:
        warnings.warn("Conflicting order of the brain regions. It's suggested to use 'sort_by_ontology'.", UserWarning, stacklevel=2)
        return vs_obj # TODO: should it display a warning/debug message?

def search_file_or_simlink(file_path: str|Path) -> Path:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.exists(): # follow_symlinks=False, from Python 3.12
        if platform.system() == "Windows":
            file_path_lnk = file_path.with_suffix(file_path.suffix + ".lnk")
            if file_path_lnk.exists(): # follow_symlinks=True, from Python 3.12
                try:
                    return Path(resolve_symlink(file_path_lnk))
                except OSError:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    try:
        return Path(resolve_symlink(file_path))
    except OSError:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

def save_csv(df: pd.DataFrame, output_path: Path|str, file_name: str, overwrite=False, sep="\t", decimal=".", **kwargs) -> Path:
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path/file_name
    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"The file {file_name} already exists in {output_path}!")
        else:
            print(f"WARNING: The file {file_name} already exists in {output_path}. Overwriting!")
    df.to_csv(file_path, sep=sep, decimal=decimal, mode="w", **kwargs)
    return file_path

def nrange(bottom, top, n):
    step = (abs(bottom)+abs(top))/(n-1)
    return np.arange(bottom, top+step, step)

def get_indices_where(where):
    rows = where.index[where.any(axis=1)]
    return [(row, col) for row in rows for col in where.columns if where.loc[row, col]]

def decorate_whatever(decorator):
    """
    Its a decorator for decorators:
    if 'decorator' is applied to a class object, it applies it to the __init__ function,
    else it is applied directly to the object (i.e. the function it decorates)
    """
    def updated_decorator(instance):
        if isinstance(instance, type): # it's a class
            func = instance.__init__
        else: # it's a function
            func = instance
        func.__name__ = instance.__name__
        decorated_func = decorator(func)
        if isinstance(instance, type): # it's a class
            instance.__init__ = decorated_func
            return instance
        else: # it's a function
            return decorated_func
    return updated_decorator

def _deprecated_message_params(func: Callable,
                               args: Sequence,
                               kwargs: dict,
                               deprecated_params: Collection[str],
                               since: str,
                               alternatives: dict[str,str]):
    bindings = inspect.signature(func).bind(*args, **kwargs)
    for deprecated_name in deprecated_params:
        if deprecated_name not in bindings.arguments:
            continue
        warning_message = f"'{deprecated_name}' is deprecated since {since} and may be removed in future versions."
        if alternatives is not None and deprecated_name in alternatives:
            warning_message += f" Use '{alternatives[deprecated_name]}', instead."
        warnings.warn(warning_message, category=DeprecationWarning, stacklevel=3)

def _deprecated_message_func(func: Callable,
                             since: str,
                             message: str,
                             alternatives: list[str]):
    warning_message = f"{func.__name__} is deprecated since {since} and may be removed in future versions."
    if alternatives:
        warning_message += f" Use any of the following alternatives, instead: '{'\', \''.join(alternatives)}'."
    if message:
        warning_message += " "+message
    warnings.warn(warning_message, category=DeprecationWarning, stacklevel=3)

def deprecated(*,
               since: str,
               params: list[str]=None,
               message: str=None,
               alternatives: list[str]|dict[str,str]=None):
    if params is None or len(params) == 0:
        params = []
    elif alternatives is None:
        pass
    else: # some deprecated params are specified
        if message:
            warnings.warn("'message' argument is ignored, if 'param' is specified.", SyntaxWarning, stacklevel=2)
        if len(alternatives) != 0 and not isinstance(alternatives, dict):
            raise TypeError(f"'alternatives' argument must be a dictionary, if 'param' is specified too. Not '{type(alternatives)}'")
        for param in alternatives.keys():
            if param not in params:
                raise ValueError(f"No deprecated parameter found: '{param}'")
    @decorate_whatever
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(params) > 0:
                _deprecated_message_params(func, args, kwargs, params, since, alternatives)
            else:
                _deprecated_message_func(func, since, message, alternatives)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class classproperty:
    def __init__(self, func):
        self.fget = func
    def __get__(self, instance, owner):
        return self.fget(owner)

@deprecated(since="1.1.0", alternatives=["braian.utils.resource"])
def get_resource_path(resource_name: str) -> Path:
    return resource(resource_name)

def resource(name: str) -> Path:
    with resources.as_file(resources.files(__package__)
                                    .joinpath("resources")
                                    .joinpath(name)) as path:
        return path

def _same_regions(rs: Collection[str], *others: Collection[str]) -> bool:
    rs = set(rs)
    return all(
        map(lambda other: len(rs.symmetric_difference(set(other))) == 0,
            others))

def _compatibility_check_sliced(xs: Collection,
                                *,
                                min_count: int=1,
                                check_atlas=True,
                                check_marker=True,
                                check_is_split=True,
                                check_regions: bool=False) -> bool:
    return _compatibility_check(xs, min_count=min_count,
                                check_atlas=check_atlas,
                                check_metrics=False,
                                check_marker=check_marker,
                                check_is_split=check_is_split,
                                check_hemispheres=False,
                                check_regions=check_regions)

def _compatibility_check_bd(xs: Collection,
                            *,
                            min_count: int=1,
                            check_atlas=True,
                            check_metrics=True,
                            check_hemisphere: bool=True,
                            check_regions: bool=False) -> bool:
    return _compatibility_check(xs, min_count=min_count,
                                check_atlas=check_atlas,
                                check_metrics=check_metrics,
                                check_marker=False,
                                check_is_split=False,
                                check_hemispheres=check_hemisphere,
                                check_regions=check_regions)

def _compatibility_check(xs: Collection,
                         *,
                         min_count: int=1,
                         check_atlas: bool=True,
                         check_metrics: bool=True,
                         check_marker: bool=True,
                         check_is_split: bool=True,
                         check_hemispheres: bool=True,
                         check_regions: bool=False):
    """
    Parameters
    ----------
    xs
        A collection of objects with the following attributes:
        `metric`, `is_split`, `markers` and `hemispheres`.

    Raises
    ------
    ValueError
        If `xs` is contains elements not compatible between each other.
    """
    if len(xs) < min_count:
        raise ValueError("No data available.")
    if check_atlas:
        atlas = xs[0].atlas
        if not all(atlas == x.atlas for x in xs[1:]):
            raise ValueError("Incompatible atlas")
    if check_metrics:
        metrics = set(x.metric for x in xs)
        if len(metrics) != 1:
            raise ValueError(f"Multiple metrics found: {', '.join(map(repr, metrics))}")
    if check_is_split:
        is_split = xs[0].is_split
        if not all(is_split == x.is_split for x in xs[1:]):
            raise ValueError("Incompatible hemispheric distinction")
    if check_marker:
        markers = set(xs[0].markers)
        if not all(markers == set(x.markers) for x in xs[1:]):
            raise ValueError("Different markers found")
    if check_hemispheres:
        if "hemispheres" not in xs[0].__dir__():
            if not all(xs[0].hemisphere is x.hemisphere for x in xs[1:]):
                raise ValueError("Data is from different hemispheres")
        else:
            hemispheres = set(xs[0].hemispheres)
            if not all(hemispheres == set(x.hemispheres) for x in xs[1:]):
                    raise ValueError("Data is from different hemispheres")
        if check_regions:
            if "hemiregions" in xs[0].__dir__():
                for hem,regions in xs[0].hemiregions.items():
                    if not _same_regions(regions, map(lambda x: x.hemiregions[hem], xs[1:])):
                        raise ValueError("Data is from different brain structures")
            else:
                if not _same_regions(*map(lambda x: x.regions, xs)):
                    raise ValueError("Data is from different brain structures")