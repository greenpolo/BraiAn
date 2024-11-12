import braian.stats as bas
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from braian.animal_group import AnimalGroup
from braian.brain_data import BrainData
from braian.experiment import Experiment
from braian.plot.generic import bar_sample
from braian.ontology import AllenBrainOntology
from collections.abc import Collection, Sequence
from plotly.subplots import make_subplots

__all__ = [
    "xmas_tree",
]

def xmas_tree(groups: Experiment|Collection[AnimalGroup],
              selected_regions: Collection[str],
              marker1: str, marker2: str=None,
              brain_ontology: AllenBrainOntology=None,
              pls_n_permutation: int=None, pls_n_bootstrap: int=None,
              pls_threshold: float=None, pls_seed: int=None,
              markers_salience_scores: dict[str, BrainData]=None,
              plot_scatter: bool=True,
              scatter_width: float=0.7, space_between_markers: float=0.02,
              groups_marker1_colours: Sequence=["LightCoral", "SandyBrown"],
              groups_marker2_colours: Sequence=["IndianRed", "Orange"],
              max_value: int=None,
              color_heatmap: str="deep_r", width: int=None, height: int=None) -> go.Figure:
    """
    Plots the XMasTree of the given data. This is a visualisation of whole-brain data from multiple groups
    in a way that is comprehensive and complete.

    The data is divided in two main plots: a heatmap (the trunk) and a scatter plot (the leaves and the xmas baubles).
    Both are aligned on the y-axis so that each row represents the data of a brain region across animals, groups and markers.

    If `pls_n_permutation` and `pls_n_bootstrap`—or, alternatively, `markers_salience_scores`—are specified, it dims out the
    brain regions that are not salient in a [partial least squared analysis][braian.stats.PLS] between the given `groups`.
    it is only supported when there are only two groups. 

    Parameters
    ----------
    groups
        The groups to display in the plot.
    selected_regions
        The acronyms of the brain regions shown in the plot.\\
        If `brain_ontology` is not given, selecting any missing region from the `groups` data will result in an error.
    marker1
        The name of the marker's data to plot.
    marker2
        If specified, the name of the second marker's data to plot.
    brain_ontology
        If specified, the `selected_regions` are checked against the ontology and sorted by
        [major divisions][braian.AllenBrainOntology.get_corresponding_md]. If a brain region is missing from `groups`
        but present in `selected_regions`, it is shown with all-[NA][pandas.NA] values.
    pls_n_permutation
        If specified, it corresponds to the parameter used for defining
        [how generizable the salience scores are][braian.stats.PLS.random_permutation].
    pls_n_bootstrap
        If specified, it corresponds to the parameter used for
        [finding the stable regions][braian.stats.PLS.bootstrap_salience_scores].
    pls_threshold
        The threshold used on the salience scores to define which regions are salient and which ones are not.\\
        If not specified, it applies a threshold of $~1.96$.
    pls_seed
        The random seed used for PLS's permutation and bootstrap operations.
        If specified, the salient regions are always deterministic. 
    markers_salience_scores
        The salience scores computed on `marker1` and, eventually, `marker2`.\\
        If specified, it does not use `pls_n_permutation` and `pls_bootstrap` parmeters and select the brain regions
        to dim out based on this dictonary.
    plot_scatter
        If False, it does not plot the scatter plot of the brain regions' values for each animal.
        It only plots the mean of the whole group.
    scatter_width
        The ratio of the whole plot's width dedicated to the scatter plot. The remaining 1-scatter_width is occupied by the heatmap.
    space_between_markers
        The retio of the whole plot's width dedicated to the gap between markers and used to specify the major divisions.\\
        If `marker2` and `brain_ontology` are not specified, it is not used. 
    groups_marker1_colours
        The list of colours used to identify `marker1` scatter data of each group.
    groups_marker2_colours
        The list of colours used to identify `marker2` scatter data of each group.
    max_value
        If specified, it caps the visualization of the brain data to this value. 
    color_heatmap
        The colormap used to display the data in the heatmap.
    width
        The width of the plot.
    height
        The height of the plot.

    Returns
    -------
    :
        A Plotly figure.
    """
    if isinstance(groups, Experiment):
        groups = tuple(g for g in groups.groups)
    assert 0 < scatter_width < 1, "Expecting 0 < scatter_width < 1"
    assert 0 <= space_between_markers <= 0.5, "Expecting 0 < scatter_width < 0.5"
    assert len(groups) >= 1, "You must provide at least one group!"
    # NOTE: if the groups have the same animals (i.e. same name), the heatmaps overlap

    if brain_ontology is not None:
        groups = [group.sort_by_ontology(brain_ontology, fill_nan=True, inplace=False) for group in groups]
        regions_mjd = brain_ontology.get_corresponding_md(*selected_regions)
        selected_regions = list(regions_mjd.keys())
    # elif len(groups) > 1:
    #     assert all(set(groups[0].regions) == set(group.regions) for group in in groups[1:])

    heatmap_width = 1-scatter_width
    bar_to_heatmap_ratio = np.array([heatmap_width, scatter_width])
    pls_params = dict(selected_regions=selected_regions, brain_ontology=brain_ontology,
                      plot_scatter=plot_scatter,
                      pls_n_bootstrap=pls_n_bootstrap, pls_n_permutation=pls_n_permutation,
                      pls_threshold=pls_threshold, pls_seed=pls_seed,
                      markers_salience_scores=markers_salience_scores)
    heatmaps, group_seps, bars, _max_value = marker_traces(groups, marker1, groups_marker1_colours, **pls_params)
    if marker2 is None:
        fig = prepare_subplots(1, bar_to_heatmap_ratio, space_between_markers if brain_ontology is not None else 0)
        data_range = (0, _max_value if max_value is None else max_value)

        major_divisions_subplot = 1
        units = f"{str(groups[0].metric)} [{groups[0].mean[marker1].units}]"

        fig.add_traces(heatmaps, rows=1, cols=2)
        [fig.add_vline(x=x, line_color="white", row=1, col=2) for x in group_seps]
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.add_traces(bars, rows=1, cols=3)
        fig.update_xaxes(title=units, range=data_range, row=1, col=3)
    else:
        m1_heatmaps, m1_group_seps, m1_bars, m1_max_value = heatmaps, group_seps, bars, _max_value
        m2_heatmaps, m2_group_seps, m2_bars, m2_max_value = marker_traces(groups, marker2, groups_marker2_colours, **pls_params)
        fig = prepare_subplots(2, bar_to_heatmap_ratio, space_between_markers)
        data_range = (0, max(m1_max_value, m2_max_value) if max_value is None else max_value)

        # MARKER1 - left side
        units = f"{str(groups[0].metric)} [{groups[0].mean[marker1].units}]"
        fig.add_traces(m1_bars, rows=1, cols=1)
        fig.update_xaxes(title=units, range=data_range[::-1], row=1, col=1) # NOTE: don't use autorange='(min) reversed', as it doesn't play nice with range
        fig.add_traces(m1_heatmaps, rows=1, cols=2)
        [fig.add_vline(x=x, line_color="white", row=1, col=2) for x in m1_group_seps]
        fig.update_xaxes(tickangle=45,  row=1, col=2)

        major_divisions_subplot = 3

        # MARKER2 - right side
        units = f"{str(groups[0].metric)} [{groups[0].mean[marker2].units}]"
        fig.add_traces(m2_heatmaps, rows=1, cols=4)
        [fig.add_vline(x=x, line_color="white", row=1, col=4) for x in m2_group_seps]
        fig.update_xaxes(tickangle=45,  row=1, col=4)
        fig.add_traces(m2_bars, rows=1, cols=5)
        fig.update_xaxes(title=units, range=data_range, row=1, col=5)

    if brain_ontology is not None:
        # add a fake trace to the empty subplot, otherwise add_annotation yref="y" makes no sense
        fig.add_trace(go.Scatter(x=[None], y=[selected_regions[len(selected_regions)//2]], mode="markers", name=None, showlegend=False), row=1, col=major_divisions_subplot)
        regions = list(regions_mjd.keys())
        mjds = np.asarray(list(regions_mjd.values()))
        prev = 0
        for y in itertools.chain(np.where(mjds[:-1] != mjds[1:])[0], (len(mjds)-1,)):
            fig.add_hline(y=y+.5, line_color="white")
            mjd = mjds[y]
            n_of_mjd = (y-prev)
            middle_of_mjd = regions[prev+(n_of_mjd//2)] if n_of_mjd != 1 else regions[y]
            fig.add_annotation(x=0, y=middle_of_mjd, text=f"<b>{mjd}</b>", showarrow=False, font_size=15, textangle=90, align="center", xanchor="center", yanchor="middle", row=1, col=major_divisions_subplot)
            prev = y
        fig.update_xaxes(showticklabels=False, row=1, col=major_divisions_subplot)
    elif marker2 is None:
        # the major division's subplot is not used, so we have to use the ticks of the heatmap
        fig.update_yaxes(showticklabels=True, row=1, col=2)

    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed") #, title="region")
    fig.update_layout(height=height, width=width, plot_bgcolor="rgba(0,0,0,0)", legend=dict(tracegroupgap=0), scattermode="group")
    fig.update_coloraxes(colorscale=color_heatmap, cmin=data_range[0], cmax=data_range[1],
                         colorbar=dict(lenmode="fraction", len=1-scatter_width+.03, thickness=15, outlinewidth=1,
                                       orientation="h", yref="container", y=1, ypad=0,
                                       title=(units if marker2 is None else units.replace(marker2, "marker"))+"\n",
                                       title_side="top"))
    return fig

def prepare_subplots(n_markers: int, bar_to_heatmap_ratio: float, gap_width: float) -> go.Figure:
    available_plot_width = (1-gap_width)/n_markers
    marker_ratio = bar_to_heatmap_ratio*available_plot_width
    if n_markers == 1:
        column_widths = [gap_width, *marker_ratio] # 3 subplots, with the first one being a spacer
    elif n_markers == 2:
        column_widths = [*marker_ratio[::-1], gap_width, *marker_ratio] # 5 subplots, with the middle one being a spacer
    else:
        raise ValueError("Cannot create a gridplot for more than 2 markers")
    fig = make_subplots(rows=1, cols=(2*n_markers)+1, horizontal_spacing=0, column_widths=column_widths, shared_yaxes=True)
    return fig


def marker_traces(groups: list[AnimalGroup],
                  marker: str, groups_colours: list,
                  selected_regions: Collection[str],
                  plot_scatter: bool,
                  brain_ontology: AllenBrainOntology,
                  pls_n_bootstrap: int, pls_n_permutation: int,
                  pls_threshold: float, pls_seed: int,
                  markers_salience_scores: dict[str, BrainData]):
    for group in groups:
        assert marker in group.markers, f"Missing {marker} in {group}"
    metric = str(groups[0].metric)
    for group in groups[1:]:
        assert str(group.metric) == metric, f"Expected metric for {group} is '{metric}'"
    assert len(groups_colours) >= len(groups), f"{marker}: You must provide a colour for each group!"
    groups_df = [group.to_pandas(marker=marker).loc[selected_regions] for group in groups] # .loc sorts the DatFrame in selected_regions' order
    if pls_filtering:=len(groups) == 2 and pls_n_bootstrap is not None and pls_n_permutation is not None:
        if markers_salience_scores is None:
            salience_scores = bas.pls_regions_salience(groups[0], groups[1], selected_regions, marker=marker, fill_nan=True,
                                                    n_bootstrap=pls_n_bootstrap, n_permutation=pls_n_permutation, seed=pls_seed)
        else:
            salience_scores =  markers_salience_scores[marker]
        if brain_ontology is not None:
            salience_scores = salience_scores.sort_by_ontology(brain_ontology, fill_nan=False, inplace=False).data
        else:
            salience_scores = salience_scores.data
        assert len(salience_scores) == len(groups_df[0]) and all(salience_scores.index == groups_df[0].index), \
                f"The salience scores of the PLS on '{marker}' are on different regions/order. "+\
                "Make sure to fill to NaN the scores for the regions missing in at least one animal."
        threshold = bas.PLS.to_zscore(p=0.05, two_tailed=True) if pls_threshold is None else pls_threshold
    # bar_sample() returns 2(+1) traces: a real one, one for the legend and, eventually, a scatter plot
    bars = [trace for group, group_df, group_colour in zip(groups, groups_df, groups_colours)
                    for trace in (bar_sample(group_df, group.name, metric, marker, group_colour, plot_scatter, plot_hash=group.name,
                                            salience_scores=salience_scores, threshold=threshold) if pls_filtering
                            else bar_sample(group_df, group.name, metric, marker, group_colour, plot_scatter, plot_hash=group.name))]
    # heatmap() returns 2 traces: a real one and one for NaNs
    heatmaps = [trace for group_df in groups_df for trace in heatmap(group_df, metric, marker)]
    _max_value = pd.concat((group.mean(axis=1)+group.sem(axis=1) for group in groups_df)).max()
    heatmap_group_seps = np.cumsum([group_df.shape[1] for group_df in groups_df[:-1]])-.5
    return heatmaps, heatmap_group_seps, bars, _max_value

def heatmap(group_df: pd.DataFrame, metric: str, marker: str):
    hmap = go.Heatmap(z=group_df, x=group_df.columns, y=group_df.index, hoverongaps=False, coloraxis="coloraxis", hovertemplate=heatmap_ht(marker, metric))
    if not group_df.isna().any(axis=None):
        return (hmap,)
    nan_hmap = go.Heatmap(z=pd.isna(group_df).astype(int), x=group_df.columns, y=group_df.index, hoverinfo="skip", #hoverongaps=False, hovertemplate=heatmap_ht(marker),
                        showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "silver"]])
    return hmap, nan_hmap


def heatmap_ht(marker, metric):
    return "animal: %{x}<br>region: %{y}<br>"+marker+" "+metric+": %{z:.2f}<extra></extra>"