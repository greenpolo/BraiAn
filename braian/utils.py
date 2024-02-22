import os
import numpy as np
import pandas as pd
import requests
import sys

def cache(filepath, url):
    if os.path.exists(filepath):
        return
    resp = requests.get(url)
    dir_path = os.path.dirname(os.path.realpath(filepath))
    os.makedirs(dir_path, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(resp.content)

def save_csv(df: pd.DataFrame, output_path: str, file_name:str, overwrite=False, sep="\t", decimal=".", **kwargs) -> None:
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)
    if os.path.exists(file_path):
        if not overwrite:
            raise FileExistsError(f"The file {file_name} already exists in {output_path}!")
        else:
            print(f"WARNING: The file {file_name} already exists in {output_path}. Overwriting previous CSV!")
    df.to_csv(file_path, sep=sep, decimal=decimal, mode="w", **kwargs)

def regions_to_plot(pls=None, salience_threshold=None,
                    groups: list=None, normalization: str=None, low_threshold: float=0.0, top_threshold=np.inf) -> list[str]:
    assert all((len(group.markers) == 1 for group in groups)), "Multiple markers not supported"
    if pls is not None and salience_threshold is not None:
        salience_scores = pls.above_threshold(salience_threshold)
        return list(salience_scores.index)
    elif groups is not None and normalization is not None \
        and low_threshold is not None and top_threshold is not None:
        groups_means = [group.mean[group.markers[0]] for group in groups]
        mean_sum = sum(groups_means)
        return mean_sum[(mean_sum > low_threshold) & (mean_sum < top_threshold) & (mean_sum.notnull())].sort_values().index.to_list()
    raise ValueError("You must specify either the ('pls', 'salience_threshold') or ('groups', 'normalization', 'low_threshold') parameters.")

def nrange(bottom, top, n):
    step = (abs(bottom)+abs(top))/(n-1)
    return np.arange(bottom, top+step, step)

def get_indices_where(where):
    rows = where.index[where.any(axis=1)]
    return [(row, col) for row in rows for col in where.columns if where.loc[row, col]]

def remote_dirs(experiment_dir_name: str,
                is_collaboration_project: bool,
                collaboration_dir_name: str) -> tuple[str,str]:
    match sys.platform:
        case "darwin":
            mnt_point = "/Volumes/Ricerca/"
            
        case "linux":
            mnt_point = "/mnt/tenibre/"
            # mnt_point = "/run/user/1000/gvfs/smb-share:server=ich.techosp.it,share=ricerca/"
        case "win32":
            mnt_point = r"\\sshfs\bs@tenibre.ipmc.cnrs.fr!2222\bs\ ".strip()
            # mnt_point = "\\\\ich.techosp.it\\Ricerca\\"
        case _:
            raise Exception(f"Can't find the 'Ricerca' folder in the server for '{sys.platform}' operative system. Please report the developer (Carlo)!")
    if not os.path.isdir(mnt_point):
        raise Exception(f"Could not read '{mnt_point}'. Please be sure you are connected to the server.")
    if is_collaboration_project:
        data_root  =  os.path.join(mnt_point, "collaborations", collaboration_dir_name, "data", experiment_dir_name)
        plots_root = os.path.join(mnt_point, "collaborations", collaboration_dir_name, "results", experiment_dir_name, "plots")
    else:
        data_root  =  os.path.join(mnt_point, "projects", "data", experiment_dir_name)
        plots_root = os.path.join(mnt_point, "projects", "results", experiment_dir_name, "plots")
    return data_root, plots_root