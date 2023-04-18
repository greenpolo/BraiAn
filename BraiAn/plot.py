import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd
import numpy as np

from .utils import nrange
from .pls import PLS
from .sliced_brain import merge_sliced_hemispheres
from .animal_brain import AnimalBrain
from .animal_group import AnimalGroup
from .brain_hierarchy import AllenBrainHierarchy, MAJOR_DIVISIONS

def plot_animal_group(fig: go.Figure, group: AnimalGroup, normalization: str,
                        AllenBrain: AllenBrainHierarchy, selected_regions: list[str],
                        color: str, y_offset, use_acronyms=True) -> None:
    avg = group.group_by_region(method=normalization).mean(numeric_only=True)
    sem = group.group_by_region(method=normalization).sem(numeric_only=True)
    y_axis, acronyms = pd.factorize(group.data.loc[selected_regions].index.get_level_values(0))
    if not use_acronyms:
        ticklabels = [f"{AllenBrain.full_name[acronym]}" for acronym in acronyms]
    else:
        ticklabels = [f"{acronym}" for acronym in acronyms]
    
    # Barplot (group)
    fig.add_trace(go.Bar(
                        x = avg.loc[selected_regions],
                        name = f"{group.name} mean",
                        marker_color=color,
                        error_x = dict(
                            type="data",
                            array=sem.loc[selected_regions]
                        )
                )
    )
    # Scatterplot (animals)
    regions_data = group.select(selected_regions)
    animal_names = regions_data.index.get_level_values(1)
    fig.add_trace(go.Scatter(
                        mode = "markers",
                        y = y_axis + y_offset,
                        x = regions_data[normalization],
                        name = f"{group.name} animals",
                        customdata = np.stack((animal_names, regions_data["area"]), axis=-1),
                        hovertemplate = "Animal: %{customdata[0]}<br>Area: %{customdata[1]} mm²<br>"+normalization+": %{x:.2f} "+group.get_units(normalization),
                        opacity=0.5,
                        marker=dict(
                            #color="rgb(0,255,0)",
                            color=color,
                            size=5,
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
                selected_regions: list[str], plot_title="", use_acronyms=True,
                colors=DEFAULT_PLOTLY_COLORS, width=900,
                barheight=30, bargap=0.3, bargroupgap=0.0):
    assert len(groups) > 0, "You selected zero AnimalGroups to plot."
    fig = go.Figure()

    n_groups = len(groups)
    region_axis_width = (1-bargap)/2
    group_bar_width = (1-bargap)/n_groups
    max_bar_offset = region_axis_width - group_bar_width/2
    y_offsets = nrange(-max_bar_offset, max_bar_offset, n_groups)
    for group, y_offset, color in zip(groups, y_offsets, colors):
        ticklabels = plot_animal_group(fig, group, normalization, AllenBrain, selected_regions, color, y_offset, use_acronyms=use_acronyms)
    
    # Plot major divisions
    major_divisions = AllenBrain.get_areas_major_division(*selected_regions).items()
    major_divisions = {k: v if v is not None else "root" for (k,v) in major_divisions}
    active_major_divisions = sorted(list(set(major_divisions.values())), key=UPPER_REGIONS.index)
    n_major_divisions = [(r1, sum(r1 == r2  for r2 in major_divisions.values())) for r1 in active_major_divisions]
    regions_colours = AllenBrain.get_region_colours()
    y_start = -0.5
    major_division_height = 0.03
    dist = 0.1
    for major_division, n in n_major_divisions:
        print
        fig.add_shape(
            type="rect",
            x0=-major_division_height-0.008, x1=0-0.008, y0=y_start+(dist/2), y1=y_start+n-(dist/2),
            xref="paper",
            line=dict(width=0),
            fillcolor=regions_colours[major_division],
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
        title = plot_title,
        yaxis = dict(
            tickmode="array",
            tickvals=np.arange(0,len(selected_regions)),
            ticktext=[label+"          " for label in ticklabels]
        ),
        xaxis=dict(
            title=groups[0].get_plot_title(normalization),
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

def plot_cv_above_threshold(AllenBrain, *sliced_brains_groups, cv_threshold=1, width=700, height=500) -> go.Figure:
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    brains_name = [brain.name for group in sliced_brains_groups for brain in group]
    group_lengths = [len(group) for group in sliced_brains_groups]
    n_brains_before_group = np.cumsum(group_lengths)
    n_areas_above_thr = []
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else 0
        group_cvar_brains = [AnimalBrain(sliced_brain, mode="cvar", hemisphere_distinction=False) for sliced_brain in group_slices]
        group_cvar_brains = [AnimalBrain.filter_selected_regions(brain, AllenBrain).data for brain in group_cvar_brains]

        for j, cvars in enumerate(group_cvar_brains):
            above_threshold_filter = cvars > cv_threshold
            n_areas_above_thr.append(sum(above_threshold_filter))
            # Scatterplot (animals)
            fig.add_trace(
                go.Scatter(
                    mode = "markers",
                    y = cvars[above_threshold_filter],
                    x = [n_brains_before+j]*above_threshold_filter.sum(),
                    text = cvars.index[above_threshold_filter],
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
        title = f"Coefficient of variaton of {sliced_brains_groups[0][0].marker} across brain slices > {cv_threshold}",
        
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
    brains_name = [brain.name for group in sliced_brains_groups for brain in group]
    group_lengths = [len(group) for group in sliced_brains_groups]
    n_brains_before_group = np.cumsum(group_lengths)
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else np.int64(0)
        group_summed_brains = [AnimalBrain(sliced_brain, hemisphere_distinction=False) for sliced_brain in group_slices]
        summed_brains.extend(group_summed_brains)
        colors.extend([DEFAULT_PLOTLY_COLORS[i]]*len(group_slices))
        for j, sliced_brain in enumerate(group_slices):
            n_brain = n_brains_before+j
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
            slices_density = []
            for slice in sliced_brain.slices:
                try:
                    density = slice.data.loc[region_name, slice.marker] / slice.data.loc[region_name, "area"]
                    # density = slice.data.loc[region_name, slice.marker]
                except KeyError as e:
                    print(f"WARNING: Could not find the '{region_name}' region for image '{slice.name}' of {slice.animal}")
                    continue
                slices_density.append(density)
            fig.add_trace(
                go.Scatter(
                    mode = "markers",
                    y = slices_density,
                    x = [n_brain] * len(slices_density),
                    text = [slice.name for slice in sliced_brain.slices],
                    name = f"image's {sliced_brain.marker} density",
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
            y=[brain.data.loc[region_name, "CFos"] / brain.data.loc[region_name, "area"] for brain in summed_brains],
            marker_color=colors,
            name=f"animal's {summed_brains[0].marker} density"
        )
    )

    fig.update_layout(
        title = f"{sliced_brains_groups[0][0].marker} density in '{region_name}'",
        xaxis = dict(
            tickmode = "array",
            tickvals = np.arange(len(brains_name)),
            ticktext = brains_name
        ),
        yaxis = dict(
            title = f"{sliced_brains_groups[0][0].marker}/mm²"
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

def plot_salient_regions(salient_regions: pd.DataFrame,
                            title=None, width=300, height=500):
    fig = go.Figure([
        go.Bar(
            x=salient_regions["salience_score"],
            orientation='h',
        )
    ])
    fig.update_layout(
        title=title,
        width=width, height=height,
        xaxis=dict(
            title = "Salience score"
        ),
        yaxis = dict(
            tickmode = "array",
            tickvals = np.arange(0,len(salient_regions)),
            ticktext = salient_regions["acronym"]
        ),
    )
    return fig

def plot_cross_correlation(r, p, title="", aspect_ratio=3/2, cell_height=18, min_plot_height=500):
    cell_width = cell_height*aspect_ratio
    plt_height = max(cell_height*len(r), min_plot_height)
    plt_width = max(cell_width*len(r), min_plot_height*aspect_ratio)

    stars = get_stars(p)

    fig = go.Figure(layout=dict(title=title),
                    data=go.Heatmap(
                        x=r.index,
                        y=r.columns,
                        z=r,
                        text=stars.values,
                        zmin=-1, zmax=1, colorscale="RdBu_r",
                        customdata=np.stack((p,), axis=-1),
                        hovertemplate="%{x} - %{y}<br>r: %{z}<br>p: %{customdata[0]}<extra></extra>",
                        texttemplate="%{text}",
                        #textfont={"size":20}
    ))
    fig.update_layout(width=plt_width, height=plt_height)       
    return fig

def get_stars(p):
    stars = pd.DataFrame(index=p.index, columns=p.columns)
    stars[(~p.isna()) & (p <= 0.05)  & (p > 0.01)] = "*"
    stars[(~p.isna()) & (p <= 0.01)  & (p > 0.001)] = "**"
    stars[(~p.isna()) & (p <= 0.001)] = "***"
    stars[(~p.isna()) & (p > 0.05)] = " "
    return stars