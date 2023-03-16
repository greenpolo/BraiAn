import os
import numpy as np
import pandas as pd

def save_csv(df: pd.DataFrame, output_path: str, file_name:str, overwrite=False, sep="\t") -> None:
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)
    if os.path.exists(file_path):
        if not overwrite:
            raise FileExistsError(f"The file {file_name} already exists in {output_path}!")
        else:
            print(f"WARNING: The file {file_name} already exists in {output_path}. Overwriting previous CSV!")
    df.to_csv(file_path, sep=sep, mode="w")

def regions_to_plot(pls=None, groups: list=None, normalization: str=None, low_threshold: float=0.0, top_threshold=np.inf) -> list[str]:
    if pls is not None:
        return pls.X.columns.to_list()
    elif groups is not None and normalization is not None \
        and low_threshold is not None and top_threshold is not None:
        groups_means = [group.group_by_region(method=normalization).mean(numeric_only=True) for group in groups]
        mean_sum = sum(groups_means)
        return mean_sum[(mean_sum > low_threshold) & (mean_sum < top_threshold) & (mean_sum.notnull())].sort_values().index.to_list()
    raise ValueError("You must specify either the 'pls' or ('groups', 'normalization', 'low_threshold') parameters.")

def nrange(bottom, top, n):
    step = (abs(bottom)+abs(top))/(n-1)
    return np.arange(bottom, top+step, step)