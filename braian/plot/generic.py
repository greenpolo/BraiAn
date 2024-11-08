import itertools
import matplotlib.colors as mplc
import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
import random

from braian.sliced_brain import SlicedBrain
from braian.animal_brain import AnimalBrain
from braian.animal_group import AnimalGroup, SlicedGroup
from braian.ontology import AllenBrainOntology, UPPER_REGIONS
from braian.experiment import Experiment, SlicedExperiment
from collections.abc import Iterable, Collection, Sequence
from plotly.subplots import make_subplots


__all__ = [
    "to_rgba",
    "bar_sample",
    "group",
    "pie_ontology",
    "above_threshold",
    "slice_density",
    "region_scores",
]

def to_rgba(color: str, alpha) -> str:
    r,g,b = plc.convert_to_RGB_255(mplc.to_rgb(color))
    return f"rgba({r}, {g}, {b}, {alpha})"

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
        The brain region ontology used to gather the major divisions of each brain area.
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
        groups = (brains.animals,)
        groups_names = (brains.name,)
    elif isinstance(brains, Experiment):
        metric = brains.groups[0].metric
        groups       = tuple(g.animals for g in brains.groups)
        groups_names = tuple(g.name for g in brains.groups)
    else:
        metric = brains[0].metric
        groups = (brains,)
        groups_names = (None,)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, (group,group_name) in enumerate(zip(groups,groups_names)):

        n_above = dict()
        regions_above = dict()

        for brain in group:
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
        fig.add_scatter(
            x=list(itertools.chain(*[[k]*len(v) for k,v in regions_above.items()])),
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
            name=group_name,
            legendgroup=i,
            mode="markers"
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

def slice_density(brains: SlicedExperiment|SlicedGroup|Sequence[SlicedBrain],
                        regions: Collection, width: int=700, height: int=500) -> go.Figure:
    """
    Scatter plot of the sections' density.

    Parameters
    ----------
    brains
        The brains from where to get the data.
    regions
        The regions to plot. If the data of `brains` is split between left and right hemisphere,
        you can pass, for example, both "Left: Isocortex" and "Right: Isocortex".
    width
        The width of the plot.
    height
        The height of the plot.

    Returns
    -------
    :
        A Plotly figure.
    """
    if isinstance(brains, SlicedGroup):
        groups = (brains.animals,)
        groups_names = (brains.name,)
    elif isinstance(brains, SlicedExperiment):
        groups       = tuple(g.animals for g in brains.groups)
        groups_names = tuple(g.name for g in brains.groups)
    else:
        groups = (brains,)
        groups_names = (None,)

    fig = go.Figure()
    for i, (group, group_name) in enumerate(zip(groups,groups_names)):
        brain_densities = []
        brain_densities_mean = dict()
        for brain in group:
            region_densities = {(slice,marker): slice.markers_density.loc[region, marker]
                                for slice in brain.slices
                                for marker in brain.markers
                                for region in regions
                                if region in slice.markers_density.index}
            if len(region_densities) == 0:
                continue
            brain_densities.append(region_densities)
            for marker in brain.markers:
                brain_densities_mean[f"{brain.name} ({marker})"] = np.mean([density for (s,m),density in region_densities.items() if m == marker])

        xs = [f"{slice.animal} ({marker})" for region_densities in brain_densities for slice,marker in region_densities.keys()]
        texts =                [slice.name for region_densities in brain_densities for slice,_      in region_densities.keys()]

        fig.add_bar(
                x=list(brain_densities_mean.keys()),
                y=list(brain_densities_mean.values()),
                # text=texts,
                marker_color=plc.DEFAULT_PLOTLY_COLORS[i],
                # opacity=0.3,
                # showlegend=i==0,
                # legendgroup="#above",
                name=f"{group_name} - mean"
        )
        fig.add_scatter(
                x=xs,
                y=[slice_density for region_densities in brain_densities for slice_density in region_densities.values()],
                text=texts,
                opacity=0.7,
                marker=dict(
                    size=7,
                    color=plc.DEFAULT_PLOTLY_COLORS[i],
                    line=dict(
                        color="rgb(0,0,0)",
                        width=1
                    )
                ),
                name=f"{group_name} - slices",
                mode="markers"
        )
    fig.update_layout(
        title = f"density in {list(regions)}",
        yaxis = dict(
            title = f"marker/mm²"
        ),
#        hovermode="x unified",
        width=width, height=height,
        template="none"
    )
    return fig


def region_scores(scores: pd.Series, brain_ontology: AllenBrainOntology,
                  title: str=None, title_size: int=20,
                  regions_size: int=15, use_acronyms: bool=True, use_acronyms_in_mjd: bool=True,
                  mjd_opacity: float=0.5, thresholds: float|Collection[float]=None, width: int=800,
                  barheight:float=30, bargap: float=0.3, bargroupgap: float=0.0): #, height=500):
    """
    Bar plot of the given regions' scores, visually grouped by the major divisions of the given `brain_ontology`.

    Parameters
    ----------
    scores
        A series of scores for each brain region, where each brain region is represented by its acronym and it is the index of the scores.
    brain_ontology
        The brain region ontology used to gather the major divisions of each brain area.
    title
        The title of the plot.
    title_size
        The size of the title.
    regions_size
        The size of each brain region name.
    use_acronyms
        If True, it uses the acronym of the brain regions instead of their full name.
    use_acronyms_in_mjd
        If True, it uses the acronym of the major divisions instead of their full name.
    mjd_opacity
        The amount of opacity used for the background of bar plot, delimiting each major division.
    thresholds
        If specified, it plots a vertical dotted line at the given value.
    width
        The width of the plot.

    Returns
    -------
    :
        A Plotly figure.

    See also
    --------
    [braian.AllenBrainOntology.get_corresponding_md][]
    """
    active_mjd = tuple(brain_ontology.get_corresponding_md(*scores.index).values())
    allen_colours = brain_ontology.get_region_colors()
    fig = go.Figure([
        go.Bar(
            x=scores,
            y=[
                [mjd.upper() if use_acronyms_in_mjd else brain_ontology.full_name[mjd].upper() for mjd in active_mjd],
                scores.index if use_acronyms else [brain_ontology.full_name[r] for r in scores.index]
            ],
            marker_color=[allen_colours[r] for r in scores.index],
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
            tickfont=dict(size=regions_size)
        ),
        yaxis=dict(
            autorange="reversed",
            dtick=1,
            tickfont=dict(size=regions_size)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="simple_white"
    )
    return fig