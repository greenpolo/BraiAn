import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..brain_hierarchy import AllenBrainHierarchy, UPPER_REGIONS

# used both for Structural Connectome and Functional Connectome
class ConnectomeAdjacency:
    def __init__(self, A: pd.DataFrame, AllenBrain: AllenBrainHierarchy, name="") -> None:
        self.upper_regions = AllenBrain.get_areas_major_division(*A.index)
        self.upper_regions = {k: v if v is not None else "root" for (k,v) in self.upper_regions.items()}
        self.A = self._sort_by_upper_regions(A)
        self.name = name

    def remove_nan_regions(self):
        for region in self.A.index[self.A.isna().all(axis=0)]:
            del self.upper_regions[region]
        self.A.dropna(axis=0, how="all", inplace=True)
        self.A.dropna(axis=1, how="all", inplace=True)
    
    def _sort_by_upper_regions(self, A: pd.DataFrame):
        upper_regions_list = list(self.upper_regions.values())
        regions_i = sorted(range(len(A)), key=lambda i: UPPER_REGIONS.index(upper_regions_list[i]))
        return A.iloc[regions_i].iloc[:,regions_i]

    def plot(self, title="", aspect_ratio=3/2, cell_height=18, min_plot_height=500,
             colorscale="RdBu_r", color_min=None, color_max=None, log=False):
        cell_width = cell_height*aspect_ratio
        plt_height = max(cell_height*len(self.A), min_plot_height)
        plt_width = max(cell_width*len(self.A), min_plot_height*aspect_ratio)
        if log:
            colors = np.log10(self.A.values)
            colorbar = dict(
                            title="Connection Weight<br>[log10]",
                        )
            customdata = np.stack((self.A.values,), axis=-1)
            hovertemplate = "%{x} - %{y}<br>weigth: %{customdata[0]}<br>log10(weight): %{z}<extra></extra>"
        else:
            colors = self.A.values.copy()
            colorbar = dict(
                            title="Connection Weight"
                        )
            customdata = None
            hovertemplate = "%{x} - %{y}<br>weigth: %{z}<extra></extra>"
        if color_min is None:
            color_min = colors[np.isfinite(colors)].min(axis=None)
        if color_max is None:
            color_max = colors[np.isfinite(colors)].max(axis=None)
        fig = go.Figure(layout=dict(title=title),
                        data=go.Heatmap(
                            x=self.A.index,
                            y=self.A.columns,
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