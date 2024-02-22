# Allen's normalised density connectome

import igraph as ig
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .connectome import Connectome
from .connectome_adjacency import ConnectomeAdjacency
from ..brain_hierarchy import AllenBrainHierarchy

class StructuralConnectome(Connectome):
    def __init__(self, normalized_connection_density_file: str,
                 regions: list[str], brain_onthology: AllenBrainHierarchy,
                 mode="max", name="Allen's ST - normalized density",
                 log10_cutoff=-5, weighted=True,
                 isolated_vertices=True) -> None:
        # NOTE: no check is done whether 'regions' is a good cut or not of Allen's ontology
        self.A = self.__read_adjacency_matrix(normalized_connection_density_file, regions, brain_onthology, mode, name)
        self.log10_cutoff = log10_cutoff
        self.mask = self.A.data >= 10**log10_cutoff
        A = self.A.data.copy()
        A[~self.mask] = np.nan
        super().__init__(A.fillna(0), isolated_vertices=isolated_vertices, weighted=weighted, directed=True, name=name, weight_str="Normalized connection density")
        self.__add_vertices_attributes(self.A)

    def __read_adjacency_matrix(self, normalized_connection_density_file: str,
                                regions: list[str], brain_onthology: AllenBrainHierarchy,
                                mode: str, name) -> ConnectomeAdjacency:
        normalized_connection_density = pd.read_csv(normalized_connection_density_file, index_col=0, header=[0,1])
        match mode:
            case "max":
                # use the maximum connection density between ipsi and contra
                normalized_connection_density = normalized_connection_density.groupby(axis=1, level=1).max()
            case "ipsi":
                normalized_connection_density = normalized_connection_density.ipsi
            case "contra":
                normalized_connection_density = normalized_connection_density.contra
            case _:
                raise ValueError(f"Unrecognized value '{mode}' for 'mode' parameter.")
        try:
            # select and sort the adjacency matrix based on 'regions'
            normalized_connection_density = normalized_connection_density.loc[regions, regions]
        except KeyError as e:
            missing_region = str(e).split("'")[1]
            message = f"Cannot find region '{missing_region}' in Allen's connectivity network."
            raise ValueError(message) from e
        return ConnectomeAdjacency(A=normalized_connection_density, brain_onthology=brain_onthology, name=name)

    def __add_vertices_attributes(self, M: ConnectomeAdjacency):
        for v in self.G.vs:
            v_name = v["name"]
            v["upper_region"] = M.upper_regions[v_name]

    def plot_adjacency(self, color_min=None, color_max=None, **kwargs) -> go.Figure:
        if color_min is None:
            color_min = self.log10_cutoff
        if color_max is None:
            color_max = self.A.max(log=True)
        if color_max <= color_min:
            raise ValueError(f"'color_max' ({color_max}) must be bigger than 'color_min' ({color_min})")
        title = f"Allen's adjacency matrix [log10(normalized density) >= {color_min}]"
        fig = self.A.plot(log=True, aspect_ratio=1, colorscale="Magma", color_min=color_min, color_max=color_max, title=title, **kwargs)
        # fig.update_layout(margin=dict(l=50, r=50, b=50, t=175, pad=0, autoexpand=True))
        fig.update_layout(yaxis=dict(scaleanchor="x"))
        return fig