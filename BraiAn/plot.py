import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd
import numpy as np

from .utils import nrange
from .pls import PLS
from .animal_brain import AnimalBrain, filter_selected_regions
from .animal_group import AnimalGroup
from .brain_hierarchy import AllenBrainHierarchy

def plot_animal_group(fig: go.Figure, group: AnimalGroup, normalization: str,
                        AllenBrain: AllenBrainHierarchy, selected_regions: list[str],
                        color: str, y_offset, use_acronyms=True) -> None:
    avg = group.group_by_region(col=normalization).mean(numeric_only=True)
    sem = group.group_by_region(col=normalization).sem(numeric_only=True)
    y_axis, ticklabels = pd.factorize(group.data.loc[selected_regions].index.get_level_values(0))
    if not use_acronyms:
        ticklabels = [AllenBrain.full_name[acronym] for acronym in ticklabels]
    
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
    fig.add_trace(go.Scatter(
                        mode = "markers",
                        y = y_axis + y_offset,
                        x = group.select(selected_regions)[normalization],
                        name = f"{group.name} animals",
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

def plot_groups(normalization: str, AllenBrain, *groups: list[AnimalGroup],
                selected_regions=None, use_acronyms=True,
                colors=DEFAULT_PLOTLY_COLORS,
                width=900, height=5000):
    assert len(groups) > 0, "You selected zero AnimalGroups to plot."
    title = groups[0].get_plot_title(normalization)
    fig = go.Figure()
    y_offsets = nrange(-0.2, +0.2, len(groups))
    for group, y_offset, color in zip(groups, y_offsets, colors):
        ticklabels = plot_animal_group(fig, group, normalization, AllenBrain, selected_regions, color, y_offset, use_acronyms=use_acronyms)

    # Update layout
    fig.update_layout(
        title = title,
        yaxis = dict(
            tickmode = "array",
            tickvals = np.arange(0,len(selected_regions)),
            ticktext = ticklabels
        ),
        xaxis=dict(
            title = f"{groups[0].marker} density (relative to brain)"
        ),
        width=width, height=height,
        hovermode="x unified",
        yaxis_range = [-1,len(selected_regions)+1]
    )

    return fig

def plot_cv_above_threshold(AllenBrain, *sliced_brains_groups, cv_threshold=1) -> go.Figure:
    fig = go.Figure()
    brains_name = [brain.name for group in sliced_brains_groups for brain in group]
    group_lengths = [len(group) for group in sliced_brains_groups]
    n_brains_before_group = np.cumsum(group_lengths)
    for i, group_slices in enumerate(sliced_brains_groups):
        n_brains_before = n_brains_before_group[i-1] if i > 0 else 0
        group_cvar_brains = [AnimalBrain(sliced_brain, mode="cvar", hemisphere_distinction=False) for sliced_brain in group_slices]
        group_cvar_brains = [filter_selected_regions(brain, AllenBrain).data for brain in group_cvar_brains]

        for j, cvars in enumerate(group_cvar_brains):
            above_threshold_filter = cvars > cv_threshold
            # Scatterplot (animals)
            fig.add_trace(go.Scatter(
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
                                showlegend=False
                        )
        )

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
        width=700, height=500
    )
    return fig

def plot_region_density(region_name, *sliced_brains_groups, width=700, height=500) -> go.Figure:
    sum_brains = []
    colors = []
    for i, group_slices in enumerate(sliced_brains_groups):
        group_summed_brains = [AnimalBrain(sliced_brain, hemisphere_distinction=False) for sliced_brain in group_slices]
        sum_brains.extend(group_summed_brains)
        colors.extend([DEFAULT_PLOTLY_COLORS[i]]*len(group_slices))

    fig  = go.Figure(
        go.Bar(
            y=[brain.data.loc[region_name, "CFos"] / brain.data.loc[region_name, "area"] for brain in sum_brains],
            marker_color=colors
        )
    )

    fig.update_layout(
        title = f"{sum_brains[0].marker} Density in '{region_name}'",
        xaxis = dict(
            tickmode = "array",
            tickvals = np.arange(len(sum_brains)),
            ticktext = [brain.name for brain in sum_brains]
        ),
        hovermode="x unified",
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