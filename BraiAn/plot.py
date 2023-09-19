import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd
import numpy as np

from .utils import nrange
from .pls import PLS
from .sliced_brain import merge_sliced_hemispheres, SlicedBrain
from .animal_brain import AnimalBrain
from .animal_group import AnimalGroup
from .brain_hierarchy import AllenBrainHierarchy, MAJOR_DIVISIONS

def plot_animal_group(fig: go.Figure, group: AnimalGroup, normalization: str,
                        AllenBrain: AllenBrainHierarchy, selected_regions: list[str],
                        animal_size: int, color: str, y_offset, use_acronyms=True) -> None:
    if len(group.markers) > 1:
        raise ValueError("Plotting of AnimalGroups with multiple markers isn't implemented yet")
    avg = group.group_by_region(method=normalization).mean(numeric_only=True)
    sem = group.group_by_region(method=normalization).sem(numeric_only=True)
    y_axis, acronyms = pd.factorize(group.data.loc[selected_regions].index.get_level_values(0))
    full_names = [AllenBrain.full_name[acronym] for acronym in acronyms]
    if not use_acronyms:
        ticklabels = full_names
    else:
        ticklabels = acronyms
    
    # Barplot (group)
    fig.add_trace(go.Bar(
                        x = avg.loc[selected_regions],
                        name = f"{group.name} mean",
                        customdata = np.stack((full_names, sem.loc[selected_regions]), axis=-1),
                        hovertemplate = normalization+" mean: %{x:2f}±%{customdata[1]:2f} "+group.get_units(normalization)+"<br>Region: %{customdata[0]}<br>Group: "+group.name,
                        marker_color=color,
                        error_x = dict(
                            type="data",
                            array=sem.loc[selected_regions]
                        )
                )
    )
    # Scatterplot (animals)
    regions_data = group.select(selected_regions)
    animal_regions = [AllenBrain.full_name[acronym] for acronym in regions_data.index.get_level_values(0)]
    animal_names = regions_data.index.get_level_values(1)
    fig.add_trace(go.Scatter(
                        mode = "markers",
                        y = y_axis + y_offset,
                        x = regions_data[normalization],
                        name = f"{group.name} animals",
                        customdata = np.stack((animal_regions, animal_names, regions_data["area"]), axis=-1),
                        hovertemplate = normalization+": %{x:.2f} "+group.get_units(normalization)+"<br>Area: %{customdata[2]} mm²<br>Region: %{customdata[0]}<br>Animal: %{customdata[1]}",
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

def plot_groups(normalization: str, AllenBrain: AllenBrainHierarchy, *groups: AnimalGroup,
                selected_regions: list[str], plot_title="", title_size=20,
                axis_size=15, animal_size=5, use_acronyms=True,
                colors=DEFAULT_PLOTLY_COLORS, width=900,
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
        ticklabels = plot_animal_group(fig, group, normalization, AllenBrain, selected_regions, animal_size, color, y_offset, use_acronyms=use_acronyms)
    
    # Plot major divisions
    active_mjd = tuple(AllenBrain.get_areas_major_division(*selected_regions).values())
    allen_colours = AllenBrain.get_region_colors()
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
                text=AllenBrain.full_name[major_division],
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
            title=groups[0].get_plot_title(normalization),
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

def plot_pie(selected_regions: list[str], AllenBrain: AllenBrainHierarchy,
                use_acronyms=True, hole=0.3, line_width=2, text_size=12):
    active_mjd = tuple(AllenBrain.get_areas_major_division(*selected_regions).values())
    mjd_occurrences = [(mjd, active_mjd.count(mjd)) for mjd in UPPER_REGIONS]
    allen_colours = AllenBrain.get_region_colors()
    fig = go.Figure(
                    go.Pie(
                        labels=[mjd if use_acronyms else AllenBrain.full_name[mjd] for mjd,n in mjd_occurrences if n != 0],
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

def plot_cv_above_threshold(AllenBrain, *sliced_brains_groups: list[SlicedBrain], cv_threshold=1, width=700, height=500) -> go.Figure:
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    brains_name = [f"{brain.name} ({marker})" for group in sliced_brains_groups for brain in group for marker in brain.markers]
    group_lengths = [len(group)*len(group[0].markers) for group in sliced_brains_groups] # assumes all animals of a group have the same markers
    n_brains_before_group = np.cumsum(group_lengths)
    n_areas_above_thr = []
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else 0
        group_cvar_brains = [AnimalBrain(sliced_brain, mode="cvar", hemisphere_distinction=False) for sliced_brain in group_slices]
        group_cvar_brains = [AnimalBrain.filter_selected_regions(brain, AllenBrain).data for brain in group_cvar_brains]

        for j, cvars in enumerate(group_cvar_brains):
            above_threshold_filter = cvars > cv_threshold
            n_areas_above_thr.extend(above_threshold_filter.sum(axis=0)) # adds, for all markers, the number of regions above threshold
            # Scatterplot (animals)
            for m, marker_cvar in enumerate(cvars.columns):
                marker_cvar_filter = above_threshold_filter[marker_cvar]
                fig.add_trace(
                    go.Scatter(
                        mode = "markers",
                        y = cvars[marker_cvar][marker_cvar_filter],
                        x = [n_brains_before+j+m]*marker_cvar_filter.sum(),
                        text = cvars.index[marker_cvar_filter],
                        opacity=0.7,
                        marker=dict(
                            size=7,
                            color=DEFAULT_PLOTLY_COLORS[i],
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
        group_summed_brains = [AnimalBrain(sliced_brain, hemisphere_distinction=False) for sliced_brain in group_slices]
        summed_brains.extend(group_summed_brains)
        colors.extend([DEFAULT_PLOTLY_COLORS[i]]*len(group_slices)*len(group_slices[0].markers))
        for j, sliced_brain in enumerate(group_slices):
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
            for m, marker in enumerate(sliced_brain.markers):
                n_brain = n_brains_before+j+m
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
                            color=DEFAULT_PLOTLY_COLORS[i],
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
            y=[brain.data.loc[region_name, marker] / brain.data.loc[region_name, "area"] for brain in summed_brains for marker in brain.markers],
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

def plot_salient_regions(salient_regions: pd.DataFrame, AllenBrain: AllenBrainHierarchy,
                            title=None, title_size=20,
                            axis_size=15, use_acronyms=True, use_acronyms_in_mjd=True,
                            mjd_opacity=0.5, width=300,
                            barheight=30, bargap=0.3, bargroupgap=0.0):#, height=500):
    active_mjd = tuple(AllenBrain.get_areas_major_division(*salient_regions["acronym"].values).values())
    allen_colours = AllenBrain.get_region_colors()
    fig = go.Figure([
        go.Bar(
            x=salient_regions["salience_score"],
            y=[
                [mjd.upper() if use_acronyms_in_mjd else AllenBrain.full_name[mjd].upper() for mjd in active_mjd],
                salient_regions["acronym"].values if use_acronyms else [AllenBrain.full_name[r] for r in salient_regions["acronym"]]
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