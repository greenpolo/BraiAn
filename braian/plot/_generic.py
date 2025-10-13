import itertools
import matplotlib.colors as mplc
import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
import random

from collections.abc import Iterable, Collection, Sequence
from plotly.subplots import make_subplots

from braian import AnimalBrain, AnimalGroup, AtlasOntology, BrainHemisphere, Experiment, SlicedBrain, SlicedExperiment, SlicedGroup
from braian.legacy._ontology_allen import UPPER_REGIONS

__all__ = [
    "to_rgba",
    "bar_sample",
    "group",
    "pie_ontology",
    "above_threshold",
    "slice_density",
    "slices",
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
                         name=f"{population_name} brains [{marker}]", showlegend=showlegend, #legendgroup=trace_name,
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
        marker_df = group.to_pandas(marker=marker, missing_as_nan=True).loc[BrainHemisphere.BOTH]
        if check_regions:
            try:
                selected_data: pd.DataFrame = marker_df.loc[selected_regions]
            except KeyError:
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

def pie_ontology(brain_ontology: AtlasOntology, selected_regions: Collection[str],
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
    [braian.AtlasOntology.partitioned][]
    """
    active_mjd = tuple(brain_ontology.partitioned(selected_regions, partition="major divisions", key="acronym").values())
    mjd_occurrences = [(mjd, active_mjd.count(mjd)) for mjd in UPPER_REGIONS]
    allen_colours = brain_ontology.colors()
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

def above_threshold(brains: Experiment|AnimalGroup|Sequence[AnimalBrain],
                    *,
                    threshold: float,
                    ontology: AtlasOntology, regions: Sequence[str],
                    marker: str|Iterable[str]=None,
                    width: int=700, height: int=500) -> go.Figure:
    """
    Scatter plot of the regions above a threshold. Usually used together
    with [SliceMetrics.CVAR][braian.SlicedMetric].

    Parameters
    ----------
    brains
        The brains from where to get the data.
    threshold
        The threshold above which a brain region is displayed.
    regions
        The names of the brain regions to filter from.
    marker
        The marker(s) to plot the data of.\
        If `None`, it plots the data for all available markers.
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
        metric = brains[0].metric
        groups = (AnimalGroup("", brains),)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if isinstance(marker,str):
        markers = (marker,)
    elif marker is not None: # is an Iterable
        markers = marker
    else:
        markers = pd.unique(np.array([m for g in groups for m in g.markers]))
    for marker_i,marker in enumerate(markers):
        df = pd.concat([g.select(ontology, regions=regions, fill_nan=True).to_pandas(marker=marker, missing_as_nan=True) for g in groups], axis=1)
        # df.index = df.index.get_level_values(1)
        df = df[~(df < threshold)]
        fig.add_trace(
            go.Bar(
                # x=[marker]*len(groups),
                x=df.columns,
                y=(~df.isna()).sum(),
                marker_color="lightsalmon",
                opacity=0.3,
                showlegend=marker_i==0,
                legendgroup="#count",
                name=f"#regions above {threshold}",
                offsetgroup=marker
            ),
            secondary_y=True,
        )
        fig.add_scatter(
            # x=[marker]*len(groups),
            x=list(itertools.chain(*[[brain_name]*len(df) for brain_name in df.columns])),
            y=list(itertools.chain(*[animal_values.values for animal_name, animal_values in df.items()])),
            text=list(itertools.chain(*[animal_values.index for animal_name, animal_values in df.items()])),
            opacity=0.7,
            marker=dict(
                size=7,
                color=plc.qualitative.Plotly[marker_i],
                line=dict(
                    color="rgb(0,0,0)",
                    width=1
                )
            ),
            name=marker,
            offsetgroup=marker,
            legendgroup=marker,
            mode="markers"
        )
    fig.update_layout(
            title = f"{metric} > {threshold}",
            yaxis=dict(
                title=metric,
                gridcolor="#d8d8d8",
                range=(threshold,None)
            ),
            yaxis2=dict(
                title=f"#regions above {threshold}",
                griddash="dot",
                gridcolor="#d8d8d8",
            ),
            scattermode="group",
            width=width, height=height,
            template="none",
            margin=dict(l=40, r=0, t=100, b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1,
                xanchor="left", x=0)
    )
    return fig

def slice_density(brains: SlicedExperiment|SlicedGroup|Sequence[SlicedBrain],
                  region: str,
                  width: int=None, height: int=None) -> go.Figure:
    """
    Scatter plot of the `BrainSlice` density in a region.

    Parameters
    ----------
    brains
        The brains from where to get the data.
    region
        The brain structure to plot.
    width
        The width of the plot.

        This parameter is ignored
    height
        The height of the plot.

        This parameter is ignored

    Returns
    -------
    :
        A Plotly figure.
    """
    return slices(brains,
                  region=region, hemisphere=BrainHemisphere.BOTH,
                  marker=None, as_density=True)

def slices(brains: SlicedExperiment|SlicedGroup,
           *,
           region: str,
           hemisphere: BrainHemisphere=BrainHemisphere.BOTH,
           marker: str|Sequence[str]=None,
           as_density: bool=True) -> go.Figure:
    """
    Scatter plot of the `BrainSlice` distribution in a region.

    Parameters
    ----------
    brains
        The brains from where to get the data.
    region
        The brain structure to plot.
    hemisphere
        The hemisphere of the brain region. If [`BOTH`][braian.BrainHemisphere]
        and [`BrainSlice.is_split`][braian.BrainSlice.is_split], it can may both values
        of the region found in the two brain hemispheres. Otherwise, it returns zero or one value.

    Returns
    -------
    :
        A Plotly figure.
    """
    if isinstance(brains, SlicedExperiment):
        groups: list[SlicedGroup] = brains.groups
    if marker is None:
        markers = groups[0].markers
    if isinstance(marker, str):
        markers = (marker,)
    units = "marker/mm²" if as_density else "#marker"

    fig = go.Figure()
    tickstext = []
    for group_i,group in enumerate(groups):
        for marker_i,marker in enumerate(markers):
            slices = group.region(region, hemisphere=hemisphere,
                                  metric=marker, as_density=as_density)\
                          .reset_index()
            grouped = slices.groupby(["brain"])[marker]
            means = grouped.mean()
            fig.add_trace(
                go.Bar(
                    x=means.index,
                    y=means.values,
                    marker_color=plc.qualitative.Plotly[group_i],
                    name=group.name,
                    showlegend=marker_i==0,
                    legendgroup=group.name,
                    offsetgroup=marker
                )
            )
            fig.add_scatter(
                x=slices["brain"],
                y=slices[marker],
                text=slices["slice"],
                opacity=0.7,
                mode="markers",
                marker=dict(
                    size=7,
                    color=plc.qualitative.Plotly[group_i],
                    line=dict(
                        color="rgb(0,0,0)",
                        width=1
                    )
                ),
                showlegend=False,
                legendgroup=group.name,
                offsetgroup=marker,
            )
        counts = grouped.count()
        tickstext.extend(counts.index+" (n="+counts.values.astype(str)+")")
    tickvals = list(range(len(tickstext)))
    fig.update_layout(
        scattermode="group",
        title=("density" if as_density else "marker")+f" in '{region}'",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=tickstext),
        yaxis=dict(title=units),
        template="none",
        margin=dict(l=40, r=0, t=60, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1,
            xanchor="left", x=0
        )
    )
    return fig


def region_scores(scores: pd.Series, brain_ontology: AtlasOntology,
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
    [braian.AtlasOntology.partitioned][]
    """
    active_mjd = tuple(brain_ontology.partitioned(scores.index, partition="major divisions", key="acronym").values())
    allen_colours = brain_ontology.colors()
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
