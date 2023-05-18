import functools
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import pearsonr
from typing import Self
from ..animal_group import AnimalGroup
from ..brain_hierarchy import AllenBrainHierarchy, UPPER_REGIONS

class CrossCorrelation:
    def __init__(self, animal_group: AnimalGroup, regions: list[str], AllenBrain: AllenBrainHierarchy,
                 normalization: str, min_animals: int) -> None:
        assert not min_animals or (min_animals >= 2), "Invalid minimum number of animals needed for cross correlation. It must be >= 2."
        self.upper_regions = AllenBrain.get_areas_major_division(*regions)
        self.upper_regions = {k: v if v is not None else "root" for (k,v) in self.upper_regions.items()}

        normalized_data = animal_group.get_normalized_data(normalization, regions)
        if not min_animals:
            # if None, all animals must have the region
            min_animals = len(normalized_data)
        self.r = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[0], min_periods=min_animals)
        self.p = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[1], min_periods=min_animals)
        self.r = self.sort_by_upper_regions(self.r)
        self.p = self.sort_by_upper_regions(self.p)

    def remove_insufficient_regions(self):
        for region in self.r.index[self.r.isna().all(axis=0)]:
            del self.upper_regions[region]
        self.r.dropna(axis=0, how="all", inplace=True)
        self.r.dropna(axis=1, how="all", inplace=True)
        self.p.dropna(axis=0, how="all", inplace=True)
        self.p.dropna(axis=1, how="all", inplace=True)
    
    def sort_by_upper_regions(self, A: pd.DataFrame):
        upper_regions_list = list(self.upper_regions.values())
        regions_i = sorted(range(len(A)), key=lambda i: UPPER_REGIONS.index(upper_regions_list[i]))
        return A.iloc[regions_i].iloc[:,regions_i]
    
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

    def plot(self, title="", aspect_ratio=3/2, cell_height=18, min_plot_height=500, star_size=15, colorscale="RdBu_r"):
        cell_width = cell_height*aspect_ratio
        plt_height = max(cell_height*len(self.r), min_plot_height)
        plt_width = max(cell_width*len(self.r), min_plot_height*aspect_ratio)

        stars = get_stars(self.p)

        fig = go.Figure(layout=dict(title=title),
                        data=go.Heatmap(
                            x=self.r.index,
                            y=self.r.columns,
                            z=self.r,
                            text=stars.values,
                            zmin=-1, zmax=1, colorscale=colorscale,
                            customdata=np.stack((self.p,), axis=-1),
                            hovertemplate="%{x} - %{y}<br>r: %{z}<br>p: %{customdata[0]}<extra></extra>",
                            texttemplate="%{text}",
                            textfont=dict(size=star_size)
        ))
        fig.update_layout(
            width=plt_width, height=plt_height,
            template="simple_white",
            plot_bgcolor="rgb(150,150,150)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

def get_stars(p):
    stars = pd.DataFrame(index=p.index, columns=p.columns)
    stars[(~p.isna()) & (p <= 0.05)  & (p > 0.01)] = "*"
    stars[(~p.isna()) & (p <= 0.01)  & (p > 0.001)] = "**"
    stars[(~p.isna()) & (p <= 0.001)] = "***"
    stars[(p.isna()) | (p > 0.05)] = " "
    return stars