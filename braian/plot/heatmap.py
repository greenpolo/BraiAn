from braian import BrainData
from braian.deflector import deflect

import brainglobe_heatmap as bgh
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

__all__ = [
    "heatmap",
    "CenteredColormap"
]

def heatmap(bd: BrainData,
            brain_regions: list[str],
            output_path: str, filename: str,
            n=10, depth=None,
            hem_highlighted_regions: list[list[str]]=None,
            cmin=None, ccenter=0, cmax=None, cmaps="magma_r",
            centered_cmap=False, orientation="frontal",
            show_text=True, title=None,
            ticks=None, ticks_labels=None,
            other=None) -> None:
    if other is None:
        hems = ("both",)
        # brain_data = brain_data.loc[brain_regions]
        # brain_data = brain_data[brain_data.index.isin(brain_regions)]
        # brain_data = brain_data[~brain_data.isna().all(axis=1)]
        data = (bd.select_from_list(brain_regions, fill_nan=True),)
        data_names = (bd.data_name,)
        _cmin = data[0].min()
        _cmax = data[0].max()
    else:
        hems = ("right", "left")
        data = (bd.select_from_list(brain_regions), other.select_from_list(brain_regions))
        _cmin = min(data[0].min(), data[1].min())
        _cmax = max(data[0].max(), data[1].max())
        data_names = (bd.data_name, other.data_name)
    if cmin is None:
        cmin = math.floor(_cmin)
    if cmax is None:
        cmax = math.ceil(_cmax)
    if centered_cmap:
        assert cmin < ccenter < cmax, "The provided BrainData's range does not include zero! Are you sure you need centered_cmap=True?"
        cmaps = (CenteredColormap("RdBu", cmin, ccenter, cmax),)*2
    if isinstance(cmaps, (str, mpl.colors.Colormap)):
        cmaps = (cmaps,)*2
    if hem_highlighted_regions is not None:
        if not isinstance(hem_highlighted_regions[0], list):
            # if you passed only one list, it will highlight the same brain regions in both hemispheres
            hem_highlighted_regions = [hem_highlighted_regions]*len(hems)
        all(r in brain_regions for selected_regions in hem_highlighted_regions
                               for r in selected_regions), "Some regions in 'selected_regions' are not inside 'brain_regions'!"
    else:
        hem_highlighted_regions = [[],[]]

    heatmaps = [
        bgh.Heatmap(
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
        for d,hem,cmap in zip(data, hems, cmaps)
    ]
    title = heatmaps[0].title
    units = bd.units if bd.units is not None else title
    xrange, yrange = heatmap_range(heatmaps[0])

    if depth is not None and isinstance(depth, (int, float, np.number)):
        fig, ax = plot_slice(depth, heatmaps, data_names, hems, orientation, title, units,
                             show_text, hem_highlighted_regions, xrange, yrange, ticks, ticks_labels)
        return fig
    else:
        max_depth = heatmaps[0].scene.atlas.shape_um[bgh.slicer.get_ax_idx(orientation)]
        depths = depth if depth is not None else np.linspace(1500, max_depth-1700, n, dtype=int)
        os.makedirs(output_path, mode=0o777, exist_ok=True)
        print("depths: ", end="")
        for position in depths:
            if position < 0 or position > max_depth:
                continue
            # frontal: 1500-11500
            # horizontal: 1500-6300
            # sagittal: 1500-9700
            print(f"{position:.2f}", end="  ")
            f,ax = plot_slice(position, heatmaps, data_names, hems, orientation, title, units,
                              show_text, hem_highlighted_regions, xrange, yrange, ticks, ticks_labels)

            plot_filepath = os.path.join(output_path, filename+f"_{position:05.0f}.svg")
            f.savefig(plot_filepath)
            plt.close(f)
        print()
        return

def heatmap_range(heatmap: bgh.Heatmap):
    shape_um = np.array(heatmap.scene.atlas.shape_um)
    origin = heatmap.scene.atlas.root.center
    x = np.where(heatmap.slicer.plane0.u != 0)[0][0]
    y = np.where(heatmap.slicer.plane0.v != 0)[0][0]
    x_min, y_min = -origin[[x, y]]
    x_max, y_max = (shape_um-origin)[[x, y]]
    return (x_min, x_max), (y_min, y_max)

def plot_slice(position: int, heatmaps: list[bgh.Heatmap],
               data_names: list[str], hems: list[str],
               orientation: str, title: str,
               units: str, show_text: bool, hem_highlighted_regions: list[list[str]],
               xrange: tuple[float, float], yrange: tuple[float, float],
               ticks: list[float], ticks_labels: list[str]):
    fig, ax = plt.subplots(figsize=(9, 9))
    slicer = bgh.slicer.Slicer(position, orientation, 100, heatmaps[0].scene.root) # requires https://github.com/brainglobe/brainglobe-heatmap/pull/43
    for heatmap, highlighted_regions in zip(heatmaps, hem_highlighted_regions):
        if highlighted_regions is None:
            highlighted_regions = []
        add_projections(ax, heatmap, slicer, show_text, highlighted_regions)

    if len(heatmaps) == 2:
        ax.axvline(x=sum(ax.get_xlim())/2, linestyle="--", color="black", lw=2)

    # set title
    fig.suptitle(title, x=0.5, y=0.88, fontsize=35)
    for data_name, hem in zip(data_names, reversed(hems)): # the hemispheres are flipped because the brain is cut front->back, not back->front
        x_pos = 0.5 if hem == "both" else 0.25 if hem == "left" else 0.75
        fig.text(s=data_name, fontsize=25, ha="center", x=x_pos, y=0.12)
        # ax.set_title(data_name, loc=hem if hem != "both" else "center", y=0, pad=-15)

    # style axes
    plt.xlim(*xrange)
    plt.ylim(*yrange)
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
    cbar = ax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=heatmaps[0].vmin, vmax=heatmaps[0].vmax), cmap=heatmaps[0].cmap),
        ax=ax, label=units, fraction=0.046, pad=0.04
    )
    if ticks is not None:
        cbar.ax.set_yticks(ticks, minor=True)
        if ticks_labels is not None:
            cbar.ax.set_yticklabels(ticks_labels, minor=True)
    return fig,ax

def add_projections(ax: mpl.axes.Axes, heatmap: bgh.Heatmap,
                    slicer: bgh.slicer.Slicer, show_text: bool,
                    selected_regions: list[str]):
    projected,_ = slicer.get_structures_slice_coords(heatmap.regions_meshes, heatmap.scene.root)
    for r, coords in projected.items():
        name, segment = r.split("_segment_")
        is_selected = name in selected_regions
        filled_polys = ax.fill(
            coords[:, 0],
            coords[:, 1],
            color=heatmap.colors[name],
            label=name if segment == "0" and name != "root" else None,
            linewidth=1.5 if is_selected else 0.5,
            edgecolor="black" if is_selected else (0, 0, 0, 0.5),
            zorder=-1 if name == "root" or heatmap.colors[name] == [0,0,0] else 1 if is_selected else 0,
            alpha=0.5 if name == "root" or heatmap.colors[name] == [0,0,0] else None,
        )
        if show_text and name != "root":
            (x0, y0), (x1, y1) = filled_polys[0].get_path().get_extents().get_points()
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=10, color="black")

class NormalizedColormap(mpl.colors.LinearSegmentedColormap,
                       metaclass=deflect(
                           on_attribute="cmap",
                           arithmetics=False,
                           container=False
                        )):
    def __init__(self, cmap, norm: mpl.colors.Normalize):
        if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            self.cmap = cmap
        else:
            self.cmap = plt.get_cmap(cmap)
        self.norm = norm
        # super compatibility
        self.N = self.cmap.N
        self.colorbar_extend = self.cmap.colorbar_extend

    def __call__(self, X, alpha=None, bytes=False):
        return mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap).to_rgba(X, alpha, bytes)

class CenteredColormap(NormalizedColormap):
    def __init__(self, cmap, vmin: int, vcenter: float, vmax: int):
        center = (vcenter-vmin)/(vmax - vmin)
        norm = mpl.colors.TwoSlopeNorm(center, vmin=0, vmax=1)
        super().__init__(cmap, norm)

if __name__ == "__main__":
    import pandas as pd
    data1 = pd.Series([100,200,130,np.nan,50])
    data1.index = ["Isocortex", "TH", "HY", "HPF", "not-a-region"]
    brain_data1 = BrainData(data1, name="Control", metric="Density", units="cFos/mm²")
    data2 = pd.Series([100,300,180, np.nan,50])
    data2.index = ["Isocortex", "TH", "HY", "HPF", "not-a-region"]
    brain_data1 = BrainData(data1, name="Control1", metric="Density", units="cFos/mm²")
    brain_data2 = BrainData(data2, name="Control2", metric="Density", units="cFos/mm²")
    heatmap(brain_data1, ["Isocortex", "TH", "HY", "HPF"], "/tmp/", "heatmap", n=11,
            cmin=None, cmax=None, orientation="frontal", show_text=True, other=brain_data2)