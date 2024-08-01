import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typing

from braian.brain_hierarchy import AllenBrainHierarchy, UPPER_REGIONS

# used both for Structural Connectome and Functional Connectome
class ConnectomeAdjacency:
    def __init__(self, A: pd.DataFrame, brain_onthology: AllenBrainHierarchy, name="", upper_regions: dict=None) -> None:
        if brain_onthology is not None:
            self.upper_regions = brain_onthology.get_areas_major_division(*A.index)
            self.upper_regions = {k: v if v is not None else "root" for (k,v) in self.upper_regions.items()}
        elif upper_regions is not None:
            self.upper_regions = upper_regions
        else:
            raise ValueError("You must specify either 'brain_onthology' or 'upper_regions'")
        self.data = self._sort_by_upper_regions(A)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.__data_log10 = np.log10(self.data)
            self.__data_log10[self.data == 0] = np.NINF
        self.name = name
    
    def mask(self, where: pd.DataFrame) -> typing.Self:
        if where.shape != self.data.shape:
            raise ValueError(f"Incompatible mask shape. Expected a {self.data.shape} shape, instead got a {where.shape} shape")
        A_masked = self.data.copy()
        A_masked[~where] = np.nan
        return ConnectomeAdjacency(A_masked, None, name=self.name+" (masked)", upper_regions=self.upper_regions)

    def max(self, log=False):
        data = self.__data_log10[np.isfinite(self.__data_log10)] if log else self.data
        return np.nanmax(data, axis=None)
    
    def min(self, log=False):
        data = self.__data_log10[np.isfinite(self.__data_log10)] if log else self.data
        return np.nanmin(data, axis=None)

    def mean(self, log=False):
        data = self.__data_log10[np.isfinite(self.__data_log10)] if log else self.data
        return np.nanmean(data, axis=None)

    def std(self, log=False):
        data = self.__data_log10[np.isfinite(self.__data_log10)] if log else self.data
        return np.nanstd(data, axis=None)

    def remove_nan_regions(self):
        for region in self.data.index[self.data.isna().all(axis=0)]:
            del self.upper_regions[region]
        self.data.dropna(axis=0, how="all", inplace=True)
        self.data.dropna(axis=1, how="all", inplace=True)
    
    def _sort_by_upper_regions(self, A: pd.DataFrame):
        upper_regions_list = list(self.upper_regions.values())
        regions_i = sorted(range(len(A)), key=lambda i: UPPER_REGIONS.index(upper_regions_list[i]))
        return A.iloc[regions_i].iloc[:,regions_i]

    def plot(self, title="", aspect_ratio=3/2, cell_height=18, min_plot_height=500,
             colorscale="RdBu_r", color_min=None, color_max=None, log=False):
        cell_width = cell_height*aspect_ratio
        plt_height = max(cell_height*len(self.data), min_plot_height)
        plt_width = max(cell_width*len(self.data), min_plot_height*aspect_ratio)
        if log:
            colors = self.__data_log10.values.copy()
            colorbar=dict(title="log₁₀", len=0.5, thickness=15)
            customdata = np.stack((self.data.values,), axis=-1)
            hovertemplate = "%{x} - %{y}<br>weigth: %{customdata[0]}<br>log10(weight): %{z}<extra></extra>"
        else:
            colors = self.data.values.copy()
            colorbar=dict(title="", len=0.5, thickness=15)
            customdata = None
            hovertemplate = "%{x} - %{y}<br>weigth: %{z}<extra></extra>"
        if color_min is None:
            color_min = self.min(log=log)
        if color_max is None:
            color_max = self.max(log=log)
        fig = go.Figure(layout=dict(title=title),
                        data=go.Heatmap(
                            x=self.data.index,
                            y=self.data.columns,
                            z=colors,
                            zmin=color_min, zmax=color_max, colorscale=colorscale,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            colorbar=colorbar
        ))
        fig.update_layout(
            width=plt_width, height=plt_height,
            template="simple_white",
            plot_bgcolor="rgb(150,150,150)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig