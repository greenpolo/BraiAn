import itertools
import matplotlib.colors as mplc
import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
import random
from collections.abc import Iterable, Collection, Sequence
from plotly.subplots import make_subplots

import braian.stats as bas
from braian.utils import nrange
from braian.sliced_brain import SlicedBrain
from braian.animal_brain import AnimalBrain, SliceMetrics
from braian.animal_group import AnimalGroup
from braian.brain_data import BrainData
from braian.ontology import AllenBrainOntology, MAJOR_DIVISIONS, UPPER_REGIONS
from braian.experiment import Experiment, SlicedExperiment

__all__ = [
    "group",
    "pie_ontology",
    "above_threshold",
    "plot_region_density",
    "plot_permutation",
    "plot_groups_salience",
    #"plot_latent_component",
    "plot_latent_variable",
    "plot_salient_regions",
    "plot_gridgroups",
    "bar_sample",
    "to_rgba"
]

def group(group: AnimalGroup, selected_regions: list[str]|np.ndarray[str],
          *markers: str, colors:Iterable=[],
          orientation: str="h", check_regions: bool=True) -> go.Figure:
    """
    Scatter plot of `AnimalGroup` data in the selected brain regions.

    Parameters
    ----------
    group
        The data of a cohort to plot.
    selected_regions
        A list of the brain regions picked to plot.
    *markers
        The marker(s) to plot the data of. If not specified, it plots all markers in `group`.
    colors
        The list of colours used to identify each marker.
    orientation
        'h' for horizontal scatter plots; 'v' for vertical scatter plots.
    check_regions
        If False, it does not check whether `group` contains all `selected_regions`.
        If data for a region is missing, it will display an empty scatter.

    Returns
    -------
    :
        A Plotly figure.

    Raises
    ------
    ValueError
        If `group` has data split between left and right hemisphere.
    ValueError
        If you didn't specify a colour for one of the markers chosen to display.
    KeyError
        If at least one region in `selected_regions` is missing from `group`.
    """
    if group.is_split:
        raise ValueError("The given AnimalGroup should not have hemisphere distinction!")
    if len(markers) == 0:
        markers = group.markers
    if not isinstance(colors, Iterable) and not isinstance(colors, str):
        colors = (colors,)
    if len(colors) < len(markers):
        raise ValueError(f"You must provide at least {len(markers)} colors. One for each marker!")

    data = []
    for marker, color in zip(markers, colors):
        marker_df = group.to_pandas(marker=marker)
        if check_regions:
            try:
                selected_data: pd.DataFrame = marker_df.loc[selected_regions]
            except KeyError as e:
                raise KeyError("Could not find data for all selected brain regions.")
        else:
            selected_data: pd.DataFrame = marker_df.reindex(selected_regions)
        traces: tuple[go.Trace] = bar_sample(selected_data,
                                             group.name, str(group.metric), marker=marker,
                                             color=color, plot_scatter=True,
                                             orientation=orientation, plot_hash=None) # None -> bars are not overlapped
        data.extend(traces)
    fig = go.Figure(data=data)
    if orientation == "h":
        fig.update_xaxes(side="top")
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(legend=dict(tracegroupgap=0), scattermode="group")
    return fig

def pie_ontology(brain_ontology: AllenBrainOntology, selected_regions: Collection[str],
        use_acronyms: bool=True, hole: float=0.3, line_width: float=2, text_size: float=12) -> go.Figure:
    """
    Pie plot of the major divisions weighted on the number of corresponding selected subregions.

    Parameters
    ----------
    brain_ontology
        The brain region ontology used to gather the hierarchy of brain regions.
    selected_regions
        The selected subregions counted by major division.
    use_acronyms
        If True, it displays brain region names as acronyms. If False, it uses their full name.
    hole
        The size of the hole in the pie chart. Must be between 0 and 1. 
    line_width
        The thickness of pie's slices.
    text_size
        The size of the brain region names.

    Returns
    -------
    :
        A Plotly figure.

    See also
    --------
    [braian.AllenBrainOntology.get_corresponding_md][]
    """
    active_mjd = tuple(brain_ontology.get_corresponding_md(*selected_regions).values())
    mjd_occurrences = [(mjd, active_mjd.count(mjd)) for mjd in UPPER_REGIONS]
    allen_colours = brain_ontology.get_region_colors()
    fig = go.Figure(
                    go.Pie(
                        labels=[mjd if use_acronyms else brain_ontology.full_name[mjd] for mjd,n in mjd_occurrences if n != 0],
                        values=[n for mjd,n in mjd_occurrences if n != 0],
                        marker=dict(
                            colors=[allen_colours[mjd] for mjd,n in mjd_occurrences if n != 0],
                            line=dict(color="#000000", width=line_width)
                        ),
                        sort=False,
                        textfont=dict(size=text_size),
                        hole=hole,
                        textposition="outside", textinfo="percent+label",
                        showlegend=False
                    ))
    return fig

def above_threshold(brains: Experiment|AnimalGroup|Sequence[AnimalBrain], threshold: float,
                    regions: Sequence[str],
                    width: int=700, height: int=500) -> go.Figure:
    """
    Scatter plot of the regions above a threshold. Usually used together
    with [SliceMetrics.CVAR][braian.SliceMetrics].

    Parameters
    ----------
    brains
        The brains from where to get the data.
    threshold
        The threshold above which a brain region is displayed.
    regions
        The names of the brain regions to filter from.
    width
        The width of the plot.
    height
        The height of the plot.

    Returns
    -------
    :
        A Plotly figure.
    """
    if isinstance(brains, AnimalGroup):
        metric = brains.metric
        groups = (brains,)
    elif isinstance(brains, Experiment):
        metric = brains.groups[0].metric
        groups = brains.groups
    else:
        metric = brains[0].mode
        groups = (brains,)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, group in enumerate(groups):

        n_above = dict()
        regions_above = dict()

        for brain in group.animals:
            for marker in brain.markers:
                brain_data = brain.to_pandas(units=False)[marker].reindex(regions)
                above = brain_data > threshold
                label = f"{brain.name} ({marker})"
                n_above[label] = above.sum()
                regions_above[label] = brain_data[above]

        fig.add_trace(
            go.Bar(
                x=list(n_above.keys()),
                y=list(n_above.values()),
                marker_color="lightsalmon",
                opacity=0.3,
                showlegend=i==0,
                legendgroup="#above",
                name=f"#regions above {threshold}",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=list(itertools.chain(*[[k]*len(v) for k,v in regions_above.items()])),
                       y=list(itertools.chain(*[v.values for v in regions_above.values()])),
                       text=list(itertools.chain(*[v.index for v in regions_above.values()])),
                       opacity=0.7,
                       marker=dict(
                           size=7,
                           color=plc.DEFAULT_PLOTLY_COLORS[i],
                           line=dict(
                               color="rgb(0,0,0)",
                               width=1
                           )
                       ),
                       name=group.name,
                       legendgroup=i,
                       mode="markers")
        )
    fig.update_layout(
        title = f"{metric} > {threshold}",

        yaxis=dict(
            title=metric,
            gridcolor="#d8d8d8",
        ),
        yaxis2=dict(
            title=f"#regions above {threshold}",
            griddash="dot",
            gridcolor="#d8d8d8",
        ),
        width=width, height=height,
        template="none"
    )
    return fig

def plot_region_density(region_name, experiment: SlicedExperiment, width=700, height=500) -> go.Figure:
    summed_brains = []
    colors = []

    fig = go.Figure()
    brains_name = [f"{brain.name} ({marker})" for group in experiment.groups for brain in group.animals for marker in brain.markers]
    group_lengths = [len(group.animals)*len(group.animals[0].markers) for group in experiment.groups] # assumes all animals of a group have the same markers
    n_brains_before_group = np.cumsum(group_lengths)
    for i, group in enumerate(experiment.groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else np.int64(0)
        group_summed_brains = [AnimalBrain.from_slices(sliced_brain, hemisphere_distinction=False) for sliced_brain in group.animals]
        summed_brains.extend(group_summed_brains)
        colors.extend([plc.DEFAULT_PLOTLY_COLORS[i]]*len(group.animals)*len(group.animals[0].markers))
        for j, sliced_brain in enumerate(group.animals):
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)
            for m, marker in enumerate(sliced_brain.markers):
                n_brain = n_brains_before+(len(group.animals[0].markers)*j)+m
                slices_density = []
                for slice in sliced_brain.slices:
                    try:
                        density = slice.data.loc[region_name, marker] / slice.data.loc[region_name, "area"]
                        # density = slice.data.loc[region_name, marker]
                    except KeyError as e:
                        # print(f"WARNING: Could not find the data for marker '{marker}' in '{region_name}' region for image '{slice.name}' of {slice.animal}")
                        continue
                    slices_density.append(density)
                fig.add_trace(
                    go.Scatter(
                        mode = "markers",
                        y = slices_density,
                        x = [n_brain] * len(slices_density),
                        text = [f"{slice.name} ({marker})" for slice in sliced_brain.slices],
                        name = f"image's markers density",
                        opacity=0.7,
                        marker=dict(
                            size=7,
                            color=plc.DEFAULT_PLOTLY_COLORS[i],
                            line=dict(
                                color="rgb(0,0,0)",
                                width=1
                            )
                        ),
                        legendgroup="slices-roots",
                        showlegend=bool(n_brain==0),
                    )
                )
    fig.add_trace(
        go.Bar(
            y=[brain[marker][region_name] / brain.areas[region_name] for brain in summed_brains for marker in brain.markers],
            marker_color=colors,
            name=f"animal's markers density"
        )
    )

    fig.update_layout(
        title = f"Markers' density in '{region_name}'",
        xaxis = dict(
            tickmode = "array",
            tickvals = np.arange(len(brains_name)),
            ticktext = brains_name
        ),
        yaxis = dict(
            title = f"marker/mm²"
        ),
#        hovermode="x unified",
        width=width, height=height,
        template="none"
    )
    return fig

def plot_permutation(pls: bas.PLS, component=1) -> go.Figure:
    n,_ = pls.s_sampling_distribution.shape
    experiment = pls._s[component-1]
    permutation = pls.s_sampling_distribution
    fig = go.Figure(data=[
            go.Histogram(x=permutation[:,component-1], nbinsx=10, name=f"Sampling distribution<br>under H0 ({n} permutations)")
        ])
    fig.add_vline(x=experiment, line_width=2, line_color="red", annotation_text="Experiment")
    fig.update_layout(
            xaxis = dict(
                title="First singular value"
            ),
            yaxis=dict(
                title = "Frequency"
            ),
            width=800, height=500, showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.63
            )
        )
    return fig

def plot_groups_salience(pls: bas.PLS, component=1):
    return go.Figure(go.Bar(x=pls.u.index, y=pls.u.iloc[:,component-1]))\
                    .update_layout(title=f"Component {component}", xaxis_title="Groups")

# def plot_latent_component(pls: bas.PLS, component=1):
#     # from https://vgonzenbach.github.io/multivariate-cookbook/partial-least-squares-correlation.html#visualizing-latent-variables
#     # seems useless to me, perhaps is for different types of PLS
#     n_groups = pls.Y.shape[1]
#     scatters = []
#     for i in range(n_groups):
#         group_i = pls.Lx.index.str.endswith(str(i))
#         group_scatter = go.Scatter(x=pls.Lx.loc[group_i, component-1],
#                                    y=pls.Ly.loc[group_i, component-1],
#                                    mode="markers+text", textposition="top center",
#                                    text=pls.Lx.loc[group_i].index, name=pls.Y.columns[i]
#         )
#         scatters.append(group_scatter)
#     fig = go.Figure(scatters)
#     return fig.update_layout(title=f"Component {component}")\
#               .update_yaxes(title="Ly")\
#               .update_xaxes(title="Lx")

def plot_latent_variable(pls: bas.PLS, of="X", height=800, width=800):
    # always plots first and second components
    # of=="X" -> plots the brain scores
    # of=="Y" -> plots the group scores
    assert of.lower() in ("x", "y"), "You must choose whether to plot latent variables of X (brain scores) or of Y (group scores)"
    latent_variables = pls.Lx if of.lower() == "x" else pls.Ly
    fig = go.Figure([go.Scatter(x=latent_variables[0][pls.Y.iloc[:,i]], y=latent_variables[1][pls.Y.iloc[:,i]],
                                # mode="markers+text", textposition="top center", textfont=dict(size=8),
                                mode="markers",
                                marker=dict(size=15),
                                text=pls.Y.iloc[:,i][pls.Y.iloc[:,i]].index, name=pls.Y.columns[i])
                    for i in range(pls.Y.shape[1])])
    return fig.update_layout(template = "none", height=height, width=width)\
                .update_xaxes(title="1", zerolinecolor="#f0f0f0", gridcolor="#f0f0f0")\
                .update_yaxes(title="2", zerolinecolor="#f0f0f0", gridcolor="#f0f0f0", scaleanchor="x", scaleratio=1)

def plot_salient_regions(salience_scores: pd.Series, brain_ontology: AllenBrainOntology,
                            title=None, title_size=20,
                            axis_size=15, use_acronyms=True, use_acronyms_in_mjd=True,
                            mjd_opacity=0.5, width=300, thresholds=None,
                            barheight=30, bargap=0.3, bargroupgap=0.0):#, height=500):
    active_mjd = tuple(brain_ontology.get_corresponding_md(*salience_scores.index).values())
    allen_colours = brain_ontology.get_region_colors()
    fig = go.Figure([
        go.Bar(
            x=salience_scores,
            y=[
                [mjd.upper() if use_acronyms_in_mjd else brain_ontology.full_name[mjd].upper() for mjd in active_mjd],
                salience_scores.index if use_acronyms else [brain_ontology.full_name[r] for r in salience_scores.index]
            ],
            marker_color=[allen_colours[r] for r in salience_scores.index],
            orientation="h"
        )
    ])
    y0 = -0.5
    for mjd in UPPER_REGIONS:
        n = active_mjd.count(mjd)
        if n == 0:
            continue
        fig.add_hrect(y0=y0, y1=y0+n, fillcolor=allen_colours[mjd], line_width=0, opacity=mjd_opacity, layer="below")
        y0 += n
    if thresholds is not None:
        if isinstance(thresholds, float):
            thresholds = (thresholds,)
        for threshold in thresholds:
            fig.add_vline(threshold, opacity=1, line=dict(width=2, dash="dash", color="black"))
            fig.add_vline(-threshold, opacity=1, line=dict(width=2, dash="dash", color="black"))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=title_size)
        ),
        width=width, height=barheight*(len(active_mjd)+1), # height,
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis=dict(
            title = "Salience score",
            tickfont=dict(size=axis_size)
        ),
        yaxis=dict(
            autorange="reversed",
            dtick=1,
            tickfont=dict(size=axis_size)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="simple_white"
    )
    return fig

def plot_gridgroups(groups: list[AnimalGroup],
                    selected_regions: list[str],
                    marker1: str, marker2: str=None,
                    brain_ontology: AllenBrainOntology=None,
                    pls_n_bootstrap: int=5000,
                    pls_threshold=None, pls_seed=None,
                    markers_salience_scores: dict[str, BrainData]=None,
                    height: int=None, width: int=None, plot_scatter=True,
                    barplot_width: float=0.7, space_between_markers: float=0.02,
                    groups_marker1_colours=["LightCoral", "SandyBrown"],
                    groups_marker2_colours=["IndianRed", "Orange"],
                    max_value=None,
                    color_heatmap="deep_r") -> go.Figure:
    def heatmap_ht(marker, metric):
        return "animal: %{x}<br>region: %{y}<br>"+marker+" "+metric+": %{z:.2f}<extra></extra>"

    def heatmap(group_df: pd.DataFrame, metric: str, marker: str):
        hmap = go.Heatmap(z=group_df, x=group_df.columns, y=group_df.index, hoverongaps=False, coloraxis="coloraxis", hovertemplate=heatmap_ht(marker, metric))
        if not group_df.isna().any(axis=None):
            return (hmap,)
        nan_hmap = go.Heatmap(z=pd.isna(group_df).astype(int), x=group_df.columns, y=group_df.index, hoverinfo="skip", #hoverongaps=False, hovertemplate=heatmap_ht(marker),
                            showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "silver"]])
        return hmap, nan_hmap

    def markers_traces(groups: list[AnimalGroup], marker: str, groups_colours: list, plot_scatter: bool):
        for group in groups:
            assert marker in group.markers, f"Missing {marker} in {group}"
        metric = str(groups[0].metric)
        for group in groups[1:]:
            assert str(group.metric) == metric, f"Expected metric for {group} is '{metric}'"
        assert len(groups_colours) >= len(groups), f"{marker}: You must provide a colour for each group!"
        groups_df = [group.to_pandas(marker=marker).loc[selected_regions] for group in groups] # .loc sorts the DatFrame in selected_regions' order
        if pls_filtering:=len(groups) == 2:
            if markers_salience_scores is None:
                salience_scores = bas.pls_regions_salience(groups[0], groups[1], selected_regions, marker=marker, fill_nan=True,
                                                        n_bootstrap=pls_n_bootstrap, seed=pls_seed)
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

    def prepare_subplots(n_markers: int, bar_to_heatmap_ratio: float, gap_width: float, ) -> go.Figure:
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

    assert barplot_width < 1 and barplot_width > 0, "Expecting 0 < barplot_width < 1"
    assert len(groups) >= 1, "You must provide at least one group!"
    # NOTE: if the groups have the same animals (i.e. same name), the heatmaps overlap

    if brain_ontology is not None:
        groups = [group.sort_by_ontology(brain_ontology, fill_nan=True, inplace=False) for group in groups]
        regions_mjd = brain_ontology.get_corresponding_md(*selected_regions)
        selected_regions = list(regions_mjd.keys())
    # elif len(groups) > 1:
    #     assert all(set(groups[0].regions) == set(group.regions) for group in in groups[1:])

    heatmap_width = 1-barplot_width
    bar_to_heatmap_ratio = np.array([heatmap_width, barplot_width])
    heatmaps, group_seps, bars, _max_value = markers_traces(groups, marker1, groups_marker1_colours, plot_scatter)
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
        m2_heatmaps, m2_group_seps, m2_bars, m2_max_value = markers_traces(groups, marker2, groups_marker2_colours, plot_scatter)
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
                         colorbar=dict(lenmode="fraction", len=1-barplot_width+.03, thickness=15, outlinewidth=1,
                                       orientation="h", yref="container", y=1, ypad=0,
                                       title=(units if marker2 is None else units.replace(marker2, "marker"))+"\n",
                                       title_side="top"))
    return fig

def bar_sample(df: pd.DataFrame, population_name: str,
           metric: str, marker: str, color: str, plot_scatter: bool,
           salience_scores: pd.Series=None, threshold: float=None,
           alpha_below_thr=0.2, alpha_undefined=0.1,
           showlegend=True, orientation="h", plot_hash=None):
    # expects df to be a regions×sample DataFrame (where <rows> × <columns>)
    def bar_ht(marker, metric, base="y", length="x"):
        return f"<b>%{{meta}}</b><br>{marker} {metric}: %{{{length}}}<br>region: %{{{base}}}<br><extra></extra>"
    traces = []
    if plot_hash is None:
        plot_hash = random.random()
    if salience_scores is None:
        fill_color, line_color = color, color
    else:
        fill_color = pd.Series(np.where(salience_scores.abs().ge(threshold, fill_value=0), color, to_rgba(color, alpha_below_thr)), index=salience_scores.index)
        is_undefined = salience_scores.isna()
        fill_color[is_undefined] = to_rgba(color, alpha_undefined)
        line_color = pd.Series(np.where(is_undefined, to_rgba(color, alpha_undefined), color), index=is_undefined.index)
    trace_name = f"{population_name} [{marker}]"
    base, length = ("y", "x") if orientation == "h" else ("x", "y")
    bar = go.Bar(**{length: df.mean(axis=1), base: df.index,
                    f"error_{length}": dict(type="data", array=df.sem(axis=1), thickness=1)},
                    marker=dict(line_color=line_color, line_width=1, color=fill_color), orientation=orientation,
                    hovertemplate=bar_ht(marker, metric, base, length), showlegend=False, offsetgroup=plot_hash,
                    name=trace_name, legendgroup=trace_name, meta=trace_name)
    traces.append(bar)
    if showlegend:
        legend = go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=color, symbol="square", size=15),
                            name=trace_name, showlegend=True, legendgroup=trace_name, offsetgroup=plot_hash)
        traces.append(legend)
    if not plot_scatter:
        return tuple(traces)
    df_stacked = df.stack()
    regions = df_stacked.index.get_level_values(0)
    sample_names = df_stacked.index.get_level_values(1)
    if salience_scores is None:
        scatter_colour = fill_color
    else:
        scatter_colour = [c for c,n in zip(fill_color, (~df.isna()).sum(axis=1)) for _ in range(n)]
    scatter = go.Scatter(**{length: df_stacked, base: regions},
                         mode="markers",
                         marker=dict(color=scatter_colour, size=4, line_color="rgba(0,0,0,0.5)", line_width=1),
                         text=sample_names, hovertemplate=bar_ht(marker, metric, base, length),
                         name=f"{population_name} animals [{marker}]", showlegend=showlegend, #legendgroup=trace_name,
                         offsetgroup=plot_hash, orientation=orientation, meta=sample_names)
    traces.append(scatter)
    # scatter requires layout(scattermode="group")
    return tuple(traces)

def to_rgba(color: str, alpha) -> str:
    r,g,b = plc.convert_to_RGB_255(mplc.to_rgb(color))
    return f"rgba({r}, {g}, {b}, {alpha})"