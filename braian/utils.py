import errno
import os
import numpy as np
import pandas as pd
import platform
import requests
import sys
from importlib import resources
from pathlib import Path

from braian import resolve_symlink

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

def get_resource_path(resource_name: str):
    with resources.as_file(resources.files(__package__)
                                    .joinpath("resources")
                                    .joinpath(resource_name)) as path:
        return path


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

def silvalab_remote_dirs(experiment_dir_name: Path|str,
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
        data_root  =  mnt_point/"collaborations"/collaboration_dir_name/"data"/experiment_dir_name
        plots_root =  mnt_point/"collaborations"/collaboration_dir_name/"results"/experiment_dir_name/"plots"
    else:
        data_root  = mnt_point/"projects"/"data"/experiment_dir_name
        plots_root = mnt_point/"projects"/"results"/experiment_dir_name/"plots"
    return data_root, plots_root