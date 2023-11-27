import itertools
import matplotlib.colors as mplc
import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import nrange
from .sliced_brain import SlicedBrain
from .animal_brain import AnimalBrain
from .animal_group import AnimalGroup, PLS
from .brain_data import BrainData
from .brain_hierarchy import AllenBrainHierarchy, MAJOR_DIVISIONS

def plot_animal_group(fig: go.Figure, group: AnimalGroup,
                        brain_onthology: AllenBrainHierarchy, selected_regions: list[str],
                        animal_size: int, color: str, y_offset, marker=None, use_acronyms=True) -> None:
    if len(group.markers) > 1:
        if marker is None:
            raise ValueError("Plotting of AnimalGroups with multiple markers isn't implemented yet")
    else:
        marker = group.markers[0]
    avg = group.mean[marker].data
    sem = group.combine(pd.DataFrame.sem, numeric_only=True)[marker].data
    # selected_regions = np.asarray(group.get_regions())
    # selected_regions = selected_regions[np.isin(selected_regions, regions)]
    regions_data = group.select(selected_regions).to_pandas()
    y_axis, acronyms = pd.factorize(regions_data.index.get_level_values(0))
    full_names = [brain_onthology.full_name[acronym] for acronym in acronyms]
    if not use_acronyms:
        ticklabels = full_names
    else:
        ticklabels = acronyms
    
    # Barplot (group)
    fig.add_trace(go.Bar(
                        x = avg.loc[selected_regions],
                        name = f"{group.name} mean",
                        customdata = np.stack((full_names, sem.loc[selected_regions]), axis=-1),
                        hovertemplate = str(group.metric)+" mean: %{x:2f}±%{customdata[1]:2f} "+group.get_units(marker)+"<br>Region: %{customdata[0]}<br>Group: "+group.name,
                        marker_color=color,
                        error_x = dict(
                            type="data",
                            array=sem.loc[selected_regions]
                        )
                )
    )
    # Scatterplot (animals)
    animal_regions = [brain_onthology.full_name[acronym] for acronym in regions_data.index.get_level_values(0)]
    animal_names = regions_data.index.get_level_values(1)
    fig.add_trace(go.Scatter(
                        mode = "markers",
                        y = y_axis + y_offset,
                        x = regions_data[group.markers[0]],
                        name = f"{group.name} animals",
                        customdata = np.stack((animal_regions, animal_names, regions_data["area"]), axis=-1),
                        hovertemplate = str(group.metric)+": %{x:.2f} "+group.get_units(marker)+"<br>Area: %{customdata[2]} mm²<br>Region: %{customdata[0]}<br>Animal: %{customdata[1]}",
                        opacity=0.5,
                        marker=dict(
                            #color="rgb(0,255,0)",
                            color=color,
                            size=animal_size,
                            line=dict(
                                color="rgb(0,0,0)",
                                width=1
                            )
                        )
                )
    )
    return ticklabels

UPPER_REGIONS = ['root', *MAJOR_DIVISIONS]

def plot_groups(brain_onthology: AllenBrainHierarchy, *groups: AnimalGroup,
                selected_regions: list[str], marker=None, plot_title="", title_size=20,
                axis_size=15, animal_size=5, use_acronyms=True,
                colors=plc.DEFAULT_PLOTLY_COLORS, width=900,
                barheight=30, bargap=0.3, bargroupgap=0.0):
    # assumes that selected_regions are sorted in depth-first order of the Allen Brain's ontology
    assert len(groups) > 0, "You selected zero AnimalGroups to plot."
    fig = go.Figure()

    n_groups = len(groups)
    region_axis_width = (1-bargap)/2
    group_bar_width = (1-bargap)/n_groups
    max_bar_offset = region_axis_width - group_bar_width/2
    y_offsets = nrange(-max_bar_offset, max_bar_offset, n_groups)
    for group, y_offset, color in zip(groups, y_offsets, colors):
        ticklabels = plot_animal_group(fig, group, brain_onthology, selected_regions, animal_size, color, y_offset, marker=marker, use_acronyms=use_acronyms)
    
    # Plot major divisions
    active_mjd = tuple(brain_onthology.get_areas_major_division(*selected_regions).values())
    allen_colours = brain_onthology.get_region_colors()
    y_start = -0.5
    major_division_height = 0.03
    dist = 0.1
    for major_division in UPPER_REGIONS:
        n = active_mjd.count(major_division)
        if n == 0:
            continue
        fig.add_shape(
            type="rect",
            x0=-major_division_height-0.008, x1=0-0.008, y0=y_start+(dist/2), y1=y_start+n-(dist/2),
            xref="paper",
            line=dict(width=0),
            fillcolor=allen_colours[major_division],
            layer="below",
            name=major_division,
            label=dict(
                text=brain_onthology.full_name[major_division],
                textangle=90,
                # font=dict(size=20)
            )
        )
        y_start += n

    # Update layout
    fig.update_layout(
        title = dict(
            text=plot_title,
            font=dict(size=title_size)
        ),
        yaxis = dict(
            tickmode="array",
            tickvals=np.arange(0,len(selected_regions)),
            ticktext=[label+"          " for label in ticklabels],
            tickfont=dict(size=axis_size)
        ),
        xaxis=dict(
            title=groups[0].get_plot_title(marker),
            tickfont=dict(size=axis_size),
            rangemode="tozero",
            side="top"
        ),
        bargap=bargap,bargroupgap=bargroupgap,
        width=width, height=(barheight*n_groups+(barheight*n_groups)*bargap)*(len(selected_regions)+1), # height,
        hovermode="closest",
        # hovermode="x unified",
        yaxis_range = [-1,len(selected_regions)]
    )

    return fig

def plot_pie(selected_regions: list[str], brain_onthology: AllenBrainHierarchy,
                use_acronyms=True, hole=0.3, line_width=2, text_size=12):
    active_mjd = tuple(brain_onthology.get_areas_major_division(*selected_regions).values())
    mjd_occurrences = [(mjd, active_mjd.count(mjd)) for mjd in UPPER_REGIONS]
    allen_colours = brain_onthology.get_region_colors()
    fig = go.Figure(
                    go.Pie(
                        labels=[mjd if use_acronyms else brain_onthology.full_name[mjd] for mjd,n in mjd_occurrences if n != 0],
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

def plot_cv_above_threshold(brain_onthology, *sliced_brains_groups: list[SlicedBrain], cv_threshold=1, width=700, height=500) -> go.Figure:
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    brains_name = [f"{brain.name} ({marker})" for group in sliced_brains_groups for brain in group for marker in brain.markers]
    group_lengths = [len(group)*len(group[0].markers) for group in sliced_brains_groups] # assumes all animals of a group have the same markers
    n_brains_before_group = np.cumsum(group_lengths)
    n_areas_above_thr = []
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else 0
        group_cvar_brains = [AnimalBrain.from_slices(sliced_brain, mode="cvar", hemisphere_distinction=False) for sliced_brain in group_slices]
        group_cvar_brains = [AnimalBrain.filter_selected_regions(brain, brain_onthology) for brain in group_cvar_brains]

        for j, cvars in enumerate(group_cvar_brains):
            # Scatterplot (animals)
            for m, marker_cvar in enumerate(cvars.markers):
                marker_cvar_filter = cvars[marker_cvar].data > cv_threshold
                n_areas_above_thr.append(marker_cvar_filter.sum())
                fig.add_trace(
                    go.Scatter(
                        mode = "markers",
                        y = cvars[marker_cvar].data[marker_cvar_filter],
                        x = [n_brains_before+(len(group_slices[0].markers)*j)+m]*marker_cvar_filter.sum(),
                        text = cvars[marker_cvar].data.index[marker_cvar_filter],
                        opacity=0.7,
                        marker=dict(
                            size=7,
                            color=plc.DEFAULT_PLOTLY_COLORS[i],
                            line=dict(
                                color="rgb(0,0,0)",
                                width=1
                            )
                        ),
                        name="Regions' coefficient<br>of variation",
                        legendgroup="regions-cv",
                        showlegend=(i+j)==0
                    )
                )

    fig.add_trace(
        go.Bar(
            # x=list(range(len(n_areas_above_thr))),
            y=n_areas_above_thr,
            marker_color="lightsalmon",
            opacity=0.3,
            name=f"animals' N areas<br>above threshold",
            
        ),
        secondary_y=True,
    )
    # reoder the data: barplot below!
    fig.data = (fig.data[-1], *fig.data[:-1])

    fig.update_layout(
        title = f"Coefficient of variaton of markers across brain slices > {cv_threshold}",
        
        xaxis = dict(
            tickmode = "array",
            tickvals = np.arange(0,len(brains_name)),
            ticktext = brains_name
        ),
        yaxis=dict(
            title = "Brain regions' CV"
        ),
        width=width, height=height
    )
    return fig

def plot_region_density(region_name, *sliced_brains_groups, width=700, height=500) -> go.Figure:
    summed_brains = []
    colors = []

    fig = go.Figure()
    brains_name = [f"{brain.name} ({marker})" for group in sliced_brains_groups for brain in group for marker in brain.markers]
    group_lengths = [len(group)*len(group[0].markers) for group in sliced_brains_groups] # assumes all animals of a group have the same markers
    n_brains_before_group = np.cumsum(group_lengths)
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else np.int64(0)
        group_summed_brains = [AnimalBrain.from_slices(sliced_brain, hemisphere_distinction=False) for sliced_brain in group_slices]
        summed_brains.extend(group_summed_brains)
        colors.extend([plc.DEFAULT_PLOTLY_COLORS[i]]*len(group_slices)*len(group_slices[0].markers))
        for j, sliced_brain in enumerate(group_slices):
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)
            for m, marker in enumerate(sliced_brain.markers):
                n_brain = n_brains_before+(len(group_slices[0].markers)*j)+m
                slices_density = []
                for slice in sliced_brain.slices:
                    try:
                        density = slice.data.loc[region_name, marker] / slice.data.loc[region_name, "area"]
                        # density = slice.data.loc[region_name, marker]
                    except KeyError as e:
                        print(f"WARNING: Could not find the data for marker '{marker}' in '{region_name}' region for image '{slice.name}' of {slice.animal}")
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
        width=width, height=height
    )
    return fig

def plot_permutation(experiment, permutation, n) -> go.Figure:
    fig = go.Figure(data=[
            go.Histogram(x=permutation[:,0], nbinsx=10, name=f"Sampling distribution<br>under H0 ({n} permutations)")
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

def plot_salient_regions(salient_regions: pd.DataFrame, brain_onthology: AllenBrainHierarchy,
                            title=None, title_size=20,
                            axis_size=15, use_acronyms=True, use_acronyms_in_mjd=True,
                            mjd_opacity=0.5, width=300,
                            barheight=30, bargap=0.3, bargroupgap=0.0):#, height=500):
    active_mjd = tuple(brain_onthology.get_areas_major_division(*salient_regions["acronym"].values).values())
    allen_colours = brain_onthology.get_region_colors()
    fig = go.Figure([
        go.Bar(
            x=salient_regions["salience_score"],
            y=[
                [mjd.upper() if use_acronyms_in_mjd else brain_onthology.full_name[mjd].upper() for mjd in active_mjd],
                salient_regions["acronym"].values if use_acronyms else [brain_onthology.full_name[r] for r in salient_regions["acronym"]]
            ],
            marker_color=[allen_colours[r] for r in salient_regions["acronym"]],
            orientation="h"
        )
    ])
    y0 = -0.5
    for mjd in UPPER_REGIONS:
        n = active_mjd.count(mjd)
        if n == 0:
            continue
        fig.add_hrect(y0=y0, y1=y0+n, fillcolor=allen_colours[mjd], line_width=0, opacity=mjd_opacity)
        y0 += n

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
                    brain_onthology: AllenBrainHierarchy=None,
                    pls_n_permutations: int=5000, pls_n_bootstrap: int=5000,
                    markers_salience_scores: dict[str, BrainData]=None,
                    height: int=None, width: int=None, plot_scatter=True,
                    barplot_width: float=0.7, space_between_markers: float=0.02,
                    groups_marker1_colours=["LightCoral", "SandyBrown"],
                    groups_marker2_colours=["IndianRed", "Orange"],
                    color_heatmap="deep_r") -> go.Figure:
    
    def to_rgba(color: str, alpha) -> str:
        r,g,b = plc.convert_to_RGB_255(mplc.to_rgb(color))
        return f"rgba({r}, {g}, {b}, {alpha})"

    def bar_ht(marker, metric):
        return "<b>%{meta}</b><br>"+marker+" "+metric+": %{x}<br>region: %{y}<br><extra></extra>"
    def heatmap_ht(marker, metric):
        return "animal: %{x}<br>region: %{y}<br>"+marker+" "+metric+": %{z:.2f}<extra></extra>"

    def bar(group_df: pd.DataFrame, group_name: str, metric: str,
            marker: str, color: str, plot_scatter: bool,
            salience_scores: pd.Series=None, threshold: float=None):
        if salience_scores is None:
            fill_color, line_color = color, color
        else:
            alpha_below_thr = 0.2
            alpha_undefined = 0.1
            fill_color = pd.Series(np.where(salience_scores.abs().ge(threshold, fill_value=0), color, to_rgba(color, alpha_below_thr)), index=salience_scores.index)
            is_undefined = salience_scores.isna()
            fill_color[is_undefined] = to_rgba(color, alpha_undefined)
            line_color = pd.Series(np.where(is_undefined, to_rgba(color, alpha_undefined), color), index=is_undefined.index)
        trace_name = f"{group_name} [{marker}]"
        trace = go.Bar(x=group_df.mean(axis=1), y=group_df.index,
                        error_x=dict(type="data", array=group_df.sem(axis=1), thickness=1),
                        marker=dict(line_color=line_color, line_width=1, color=fill_color), orientation="h",
                        hovertemplate=bar_ht(marker, metric), showlegend=False, offsetgroup=group_name,
                        name=trace_name, legendgroup=trace_name, meta=trace_name)
        trace_legend = go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=color, symbol="square", size=15),
                                    name=trace_name, showlegend=True, legendgroup=trace_name, offsetgroup=group_name)
        if not plot_scatter:
            return trace, trace_legend
        group_df_ = group_df.stack()
        scatter = go.Scatter(x=group_df_, y=group_df_.index.get_level_values(0), mode="markers",
                             marker=dict(color=color, size=8, line=dict(color="rgb(0,0,0)", width=1)),
                             name=f"animal [{marker}]", showlegend=True, legendgroup=trace_name,
                             offsetgroup=group_name, orientation="h")
        return trace, trace_legend, scatter

    def heatmap(group_df: pd.DataFrame, metric: str, marker: str):
        hmap = go.Heatmap(z=group_df, x=group_df.columns, y=group_df.index, hoverongaps=False, coloraxis="coloraxis", hovertemplate=heatmap_ht(marker, metric))
        nan_hmap = go.Heatmap(z=np.isnan(group_df).astype(int), x=group_df.columns, y=group_df.index, hoverinfo="skip", #hoverongaps=False, hovertemplate=heatmap_ht(marker),
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
                salience_scores = groups[0].pls_regions(groups[1], selected_regions, marker=marker, fill_nan=True,
                                                        n_permutations=pls_n_permutations, n_bootstrap=pls_n_bootstrap)
            else:
                salience_scores =  markers_salience_scores[marker]
            if brain_onthology is not None:
                salience_scores = salience_scores.sort_by_onthology(brain_onthology, fill=False, inplace=False).data
            else:
                salience_scores = salience_scores.data
            assert all(salience_scores.index == groups_df[0].index), \
                    f"The salience scores ofthe PLS on '{marker}' are on different regions/order. "+\
                    "Make sure to fill to NaN the scores for the regions missing in at least one animal."
            threshold = PLS.norm_threshold(nsigma=3) # use the μ ± 3σ of the normal as threshold
        # bar() returns 2(+1) traces: a real one, one for the legend and, eventually, a scatter plot
        bars = [trace for group, group_df, group_colour in zip(groups, groups_df, groups_colours)
                      for trace in (bar(group_df, group.name, metric, marker, group_colour, plot_scatter, salience_scores, threshold) if pls_filtering
                               else bar(group_df, group.name, metric, marker, group_colour, plot_scatter))]
        # heatmap() returns 2 traces: a real one and one for NaNs
        heatmaps = [trace for group_df in groups_df for trace in heatmap(group_df, metric, marker)]
        max_value = pd.concat((group.mean(axis=1)+group.sem(axis=1)/2 for group in groups_df)).max()
        heatmap_group_seps = np.cumsum([group_df.shape[1] for group_df in groups_df[:-1]])-.5
        return heatmaps, heatmap_group_seps, bars, max_value
    
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

    if brain_onthology is not None:
        groups = [group.sort_by_onthology(brain_onthology, fill=True, inplace=False) for group in groups]
        regions_mjd = brain_onthology.get_areas_major_division(*selected_regions, sorted=True)
        selected_regions = list(regions_mjd.keys())

    heatmap_width = 1-barplot_width
    bar_to_heatmap_ratio = np.array([heatmap_width, barplot_width])
    heatmaps, group_seps, bars, max_value = markers_traces(groups, marker1, groups_marker1_colours, plot_scatter)
    if marker2 is None:
        fig = prepare_subplots(1, bar_to_heatmap_ratio, space_between_markers if brain_onthology is not None else 0)
        
        major_divisions_subplot = 1
        
        fig.add_traces(heatmaps, rows=1, cols=2)
        [fig.add_vline(x=x, line_color="white", row=1, col=2) for x in group_seps]
        fig.update_xaxes(tickangle=45,  row=1, col=2)
        fig.add_traces(bars, rows=1, cols=3)
    else:
        m1_heatmaps, m1_group_seps, m1_bars, m1_max_value = heatmaps, group_seps, bars, max_value
        m2_heatmaps, m2_group_seps, m2_bars, m2_max_value = markers_traces(groups, marker2, groups_marker2_colours, plot_scatter)
        fig = prepare_subplots(2, bar_to_heatmap_ratio, space_between_markers)
        bar_range = (0, max(m1_max_value, m2_max_value))
        
        # MARKER1 - left side
        fig.add_traces(m1_bars, rows=1, cols=1)
        fig.update_xaxes(autorange="reversed", range=bar_range, row=1, col=1)
        fig.add_traces(m1_heatmaps, rows=1, cols=2)
        [fig.add_vline(x=x, line_color="white", row=1, col=2) for x in m1_group_seps]
        fig.update_xaxes(tickangle=45,  row=1, col=2)

        major_divisions_subplot = 3
        
        # MARKER2 - right side
        fig.add_traces(m2_heatmaps, rows=1, cols=4)
        [fig.add_vline(x=x, line_color="white", row=1, col=4) for x in m2_group_seps]
        fig.update_xaxes(tickangle=45,  row=1, col=4)
        fig.add_traces(m2_bars, rows=1, cols=5)
        fig.update_xaxes(range=bar_range, row=1, col=5)

    if brain_onthology is not None:
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
    fig.update_layout(height=height, width=width, plot_bgcolor="rgba(0,0,0,0)", legend=dict(tracegroupgap=0), scattermode="group",
        coloraxis=dict(colorscale=color_heatmap, colorbar=dict(lenmode="pixels", len=500, thickness=15, outlinewidth=1, y=0.95, yanchor="top"))
    )
    return fig