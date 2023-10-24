import bgheatmaps as bgh
import copy
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import vedo as vd
from typing import Self

from .deflector import deflect
from .brain_hierarchy import AllenBrainHierarchy

def extract_acronym(region_class):
    '''
    This function extracts the region acronym from a QuPath's PathClass assigned by ABBA
    Example: "Left: AVA" becomes "AVA".
    '''
    acronym = re.compile("[Left|Right]: (.+)").findall(region_class)
    if len(acronym) == 0:
        # the region's class didn't distinguish between left|right hemispheres 
        return str(region_class)
    return acronym[0]

def get_hemisphere_name(hem: str):
    match hem.lower():
        case "left" | "l":
            return "Left"
        case "right" | "r":
            return "Right"
        case _:
            raise ValueError(f"Unrecognised hemisphere '{hem}'!")

def is_split_left_right(index: pd.Index):
    return (index.str.startswith("Left: ", na=False) | \
            index.str.startswith("Right: ", na=False)).all()

def split_index(regions: list[str]) -> list[str]:
    return [": ".join(t) for t in itertools.product(("Left", "Right"), regions)]

class BrainData(metaclass=deflect(on_attribute="data", arithmetics=True, container=True)):
    @staticmethod
    def merge(first: Self, second: Self, *others: Self, op=pd.DataFrame.mean, name=None, op_name=None, **kwargs) -> Self:
        assert first.metric == second.metric and all([first.metric == other.metric for other in others]), "Mean must be done between BrainData of the same metric!"
        assert first.units == second.units and all([first.units == other.units for other in others]), "Mean must be done between BrainData of the same units!"
        if name is None:
            name = first.data_name
        if op_name is None:
            op_name = op.__name__
        data = op(pd.concat([first.data, second.data, *[other.data for other in others]], axis=1), axis=1, **kwargs)
        return BrainData(data, name, f"{first.metric}-{op_name} (n={len(others)+2})", first.units) 

    @staticmethod
    def mean(*args, **kwargs) -> Self:
        return BrainData.merge(*args, op=pd.DataFrame.mean, **kwargs)

    def __init__(self, data: pd.Series, name: str, metric: str, units: str,
                 brain_onthology=None, fill=False) -> None: # brain_onthology: AllenBrainHierarchy
        self.data = data.copy()
        self.is_split = is_split_left_right(self.data.index)
        self.data_name = str(name) # data_name
        self.data.name = self.data_name
        self.metric = str(metric)
        if units is not None:
            self.units = str(units)
        else:
            self.units = ""
            print(f"WARNING: {self} has no units")
        if brain_onthology is not None:
            self.sort_by_onthology(brain_onthology, fill, inplace=True)
    
    def __str__(self) -> str:
        return f"BrainData(name={self.data_name}, metric={self.metric})"
    
    def sort_by_onthology(self, brain_onthology: AllenBrainHierarchy,
                          fill=False, inplace=False) -> Self:
        all_regions = brain_onthology.list_all_subregions("root", mode="depth")
        if self.is_split:
            all_regions = split_index(all_regions)
        if len(unknown_regions:=self.data.index[~self.data.index.isin(all_regions)]) > 0:
            raise ValueError(f"The following regions are unknown to the given brain onthology: '"+"', '".join(unknown_regions)+"'")
        if not fill:
            all_regions = np.array(all_regions)
            all_regions = all_regions[np.isin(all_regions, self.data.index)]
        # NOTE: since fill_value=np.nan -> converts dtype to float
        data = self.data.reindex(all_regions, copy=False, fill_value=np.nan)
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self
    
    def root(self, hemisphere=None) -> float:
        acronym = "root"
        if self.is_split:
            if hemisphere is None:
                raise ValueError(f"You have to specify the hemisphere of '{acronym}' you want!")
            acronym = f"{get_hemisphere_name(hemisphere)}: {acronym}"
        if acronym not in self.data:
            raise ValueError(f"No data for '{acronym}' in {self}!")
        return self.data[acronym]

    def min(self) -> float:
        return self.data[self.data != np.inf].min()

    def max(self) -> float:
        return self.data[self.data != np.inf].max()

    def remove_region(self, *region: str, inplace=False, fillnan=False) -> Self:
        data = self.data.copy() if not inplace else self.data
        if fillnan:
            data[list(region)] = np.nan
        else:
            data = data[data.index.isin(region)]
        return self if inplace else BrainData(data, name=self.data_name, metric=self.metric, units=self.units)

    def get_regions(self) -> list[str]:
        return list(self.data.index)

    def select_from_list(self, brain_regions: list[str], fill_nan=False, inplace=False) -> Self:
        if not (unknown_regions:=np.isin(brain_regions, self.data.index)).all():
            unknown_regions = np.array(brain_regions)[~unknown_regions]
            raise ValueError(f"Can't find some regions in {self}: '"+"', '".join(unknown_regions)+"'!")
        if fill_nan:
            data = self.data.reindex(index=brain_regions, fill_value=np.nan)
        else:
            data = self.data[self.data.index.isin(brain_regions)]
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self
    
    def select_from_onthology(self, brain_onthology: AllenBrainHierarchy, *args, **kwargs) -> Self:
        selected_allen_regions = brain_onthology.get_selected_regions()
        selectable_regions = set(self.data.index).intersection(set(selected_allen_regions))
        return self.select_from_list(list(selectable_regions), *args, **kwargs)
    
    def merge_hemispheres(self) -> Self:
        if self.metric not in ("sum",):
            raise ValueError(f"Cannot properly merge '{self.metric}' BrainData from left/right hemispheres into a single region!")
        corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in self.data.index]
        data = self.data.groupby(corresponding_region).sum(min_count=1)
        return BrainData(data, name=self.data_name, metric=self.metric, units=self.units)

    def plot(self,
                brain_regions: list[str],
                output_path: str, filename: str,
                n=10,
                cmin=None, cmax=None, cmap="magma_r",
                orientation="frontal",
                show_text=True, title=None,
                other=None) -> None:
        if other is None:
            hems = ("both",)
            # brain_data = brain_data.loc[selected_regions]
            # brain_data = brain_data[brain_data.index.isin(brain_regions)]
            # brain_data = brain_data[~brain_data.isna().all(axis=1)]
            data = (self.select_from_list(brain_regions),)
            data_names = (self.data_name,)
            _cmin = data[0].min()
            _cmax = data[0].max()
        else:
            hems = ("right", "left")
            data = (self.select_from_list(brain_regions), other.select_from_list(brain_regions))
            _cmin = min(data[0].min(), data[1].min())
            _cmax = max(data[0].max(), data[1].max())
            data_names = (self.data_name, other.data_name)
        if cmin is None:
            cmin = math.floor(_cmin)
        if cmax is None:
            cmax = math.ceil(_cmax)
        heatmaps = [
            bgh.heatmap(
                d.data.to_dict(),
                position=None,
                orientation=orientation,
                title=title or d.metric,
                cmap=cmap,
                vmin=cmin,
                vmax=cmax,
                format="2D",
                hemisphere=hem
            )
            for d,hem in zip(data, hems)
        ]
        title = heatmaps[0].title

        print("depths: ", end="")
        for depth in np.linspace(1500, 11000, n):
            print(f"{depth:.2f}", end="  ")
            slicer = bgh.slicer.Slicer(depth, orientation, 100, heatmaps[0].scene.root)

            f, ax = plt.subplots(figsize=(9, 9))
            for heatmap in heatmaps:
                add_projections(ax, heatmap, slicer, show_text)

            # set title
            ax.set_title(title, fontsize=20, pad=-15)
            for data_name, hem in zip(data_names, reversed(hems)): # the hemispheres are flipped because the brain is cut front->back, not back->front
                ax.set_title(data_name, loc=hem if hem != "both" else "center", y=0, pad=-15)

            # style axes
            if orientation == "frontal":
                ax.invert_yaxis()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(xlabel="", ylabel="")
            ax.set_aspect('equal',adjustable='box')

            # add colorbar
            ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=heatmaps[0].vmin, vmax=heatmaps[0].vmax), cmap=heatmaps[0].cmap),
                ax=ax, label=title, fraction=0.046, pad=0.04
            )

            plot_filepath = os.path.join(output_path, filename+f"_{depth:05.0f}.svg")
            f.savefig(plot_filepath)
            plt.close(f)
        print()

def add_projections(ax: mpl.axes.Axes, heatmap: bgh.heatmap,
                    slicer: bgh.slicer.Slicer, show_text: bool):
    projected,_ = slicer.get_structures_slice_coords(heatmap.regions_meshes, heatmap.scene.root)
    for r, coords in projected.items():
        name, segment = r.split("_segment_")
        filled_polys = ax.fill(
            coords[:, 0],
            coords[:, 1],
            color=heatmap.colors[name],
            label=name if segment == "0" and name != "root" else None,
            lw=1,
            ec="k",
            zorder=-1 if name == "root" or heatmap.colors[name] == [0,0,0] else None,
            alpha=0.3 if name == "root" or heatmap.colors[name] == [0,0,0] else None,
        )
        if show_text and name != "root":
            (x0, y0), (x1, y1) = filled_polys[0].get_path().get_extents().get_points()
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=10, color="black")

if __name__ == "__main__":
    data1 = pd.Series([100,200,130,np.nan,50])
    data1.index = ["Isocortex", "TH", "HY", "HPF", "not-a-region"]
    brain_data1 = BrainData(data1, name="Control", metric="Density", units="cFos/mm²")
    data2 = pd.Series([100,300,180, np.nan,50])
    data2.index = ["Isocortex", "TH", "HY", "HPF", "not-a-region"]
    brain_data1 = BrainData(data1, name="Control1", metric="Density", units="cFos/mm²")
    brain_data2 = BrainData(data2, name="Control2", metric="Density", units="cFos/mm²")
    brain_data1.plot(["Isocortex", "TH", "HY", "HPF"], "/tmp/", "heatmap", n=10, cmin=None, cmax=None,
                     orientation="frontal", show_text=True, other=brain_data2)