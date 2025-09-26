import errno
import functools
import igraph as ig
import itertools
import os
import numpy as np
import pandas as pd
import platform
import requests
import sys
import warnings

from collections.abc import Sequence
from importlib import resources
from pathlib import Path, WindowsPath

__all__ = ["cache", "deprecated", "get_resource_path", "resource"]

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

def merge_ordered(*xs: Sequence) -> Sequence:
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
    else:
        raise ValueError("The given sequences are not sorted in a compatible way.")
        return vs_obj

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

def decorate_all(decorator):
    # its a decorator for decorators:
    # if 'decorator' is applied to a class object, it applies it to the __init__ function
    # else, it applies it directly to the object (i.e. the function it decorates)
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

def deprecated(*, since: str, message=None, alternatives: list[str]=None):
    @decorate_all
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_message = f"{func.__name__} is deprecated since {since} and may be removed in future versions."
            if alternatives:
                warning_message += f" Use any of the following alternatives, instead: '{'\', \''.join(alternatives)}'."
            if message:
                warning_message += " "+message
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated(since="1.1.0", alternatives=["braian.utils.resource"])
def get_resource_path(resource_name: str):
    return resource(resource_name)

def resource(name: str):
    with resources.as_file(resources.files(__package__)
                                    .joinpath("resources")
                                    .joinpath(name)) as path:
        return path

def silvalab_remote_dirs(
                experiment_dir_name: Path|str,
                is_collaboration_project: bool,
                collaboration_dir_name: Path|str) -> tuple[Path,Path]:
    match sys.platform:
        case "darwin":
            mnt_point = "/Volumes/Ricerca/"

        case "linux":
            mnt_point = "/mnt/tenibre/"
            # mnt_point = "/run/user/1000/gvfs/smb-share:server=ich.techosp.it,share=ricerca/"
        case "win32":
            mnt_point = r"\\TENIBRE\bs\ ".strip()
            # mnt_point = r"\\sshfs\bs@tenibre.ipmc.cnrs.fr!2222\bs\ ".strip()
            # mnt_point = "\\\\ich.techosp.it\\Ricerca\\"
        case _:
            raise Exception(f"Can't find the 'Ricerca' folder in the server for '{sys.platform}' operative system. Please report the developer (Carlo)!")
    mnt_point = Path(mnt_point)
    if not mnt_point.is_dir():
        raise Exception(f"Could not read '{mnt_point}'. Please be sure you are connected to the server.")
    if is_collaboration_project:
        analysis_root  =  mnt_point/"collaborations"/collaboration_dir_name/experiment_dir_name/"analysis"
        plots_root =  mnt_point/"collaborations"/collaboration_dir_name/experiment_dir_name/"results"/"plots"
    else:
        analysis_root  = mnt_point/"projects"/experiment_dir_name/"analysis"
        plots_root = mnt_point/"projects"/experiment_dir_name/"results"/"plots"
    return analysis_root, plots_root