import functools
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from typing import Self
from .connectome_adjacency import ConnectomeAdjacency
from ..animal_group import AnimalGroup
from ..brain_hierarchy import AllenBrainHierarchy

class CrossCorrelation:
    def __init__(self, animal_group: AnimalGroup, regions: list[str], AllenBrain: AllenBrainHierarchy,
                 normalization: str, min_animals: int, name="") -> None:
        assert not min_animals or (min_animals >= 2), "Invalid minimum number of animals needed for cross correlation. It must be >= 2."
        normalized_data = animal_group.get_normalized_data(normalization, regions)
        self.n = len(normalized_data)
        if not min_animals:
            # if None, all animals must have the region
            min_animals = self.n
        r = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[0], min_periods=min_animals)
        p = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[1], min_periods=min_animals)
        rp_name_space = " - " if name else ""
        self.r = ConnectomeAdjacency(r, AllenBrain, name+rp_name_space+"p-value")
        self.p = ConnectomeAdjacency(p, AllenBrain, name+rp_name_space+"Pearson coefficient")
        self.name = name

    def remove_insufficient_regions(self):
        self.r.remove_nan_regions()
        self.p.remove_nan_regions()
    
    @staticmethod
    def regions_in_both_groups(cross1, cross2):
        return (~cross1.r.isna().all(axis=1)) & (~cross2.r.isna().all(axis=1))

    @staticmethod
    def make_comparable(*ccs: Self):
        regions_in_all_groups = functools.reduce(CrossCorrelation.regions_in_both_groups, ccs)
        for cc in ccs:
            cc.r.loc[~regions_in_all_groups,:] = np.nan
            cc.r.loc[:,~regions_in_all_groups] = np.nan
            cc.p.loc[:,~regions_in_all_groups] = np.nan
            cc.p.loc[~regions_in_all_groups,:] = np.nan

    def plot(self, star_size=15, **kwargs):
        fig = self.r.plot(**kwargs)
        old_customdata = fig.data[0].customdata
        # old_hovertemplate = "%{x} - %{y}<br>r: %{z}<br>p: %{customdata[0]}<extra></extra>"
        fig.update_traces(
                selector=dict(type="heatmap"),
                text=get_stars(self.p.A).values,
                customdata=np.hstack((old_customdata.customdata, np.expand_dims(self.p.A, 1))), #np.stack((self.p.A,), axis=-1),
                hovertemplate="%{x} - %{y}<br>r: %{customdata[0]}<br>p: %{customdata[1]}<extra></extra>",
                texttemplate="%{text}",
                textfont=dict(size=star_size)
            )
        return fig

def get_stars(p):
    stars = pd.DataFrame(index=p.index, columns=p.columns)
    stars[(~p.isna()) & (p <= 0.05)  & (p > 0.01)] = "*"
    stars[(~p.isna()) & (p <= 0.01)  & (p > 0.001)] = "**"
    stars[(~p.isna()) & (p <= 0.001)] = "***"
    stars[(p.isna()) | (p > 0.05)] = " "
    return stars