import braian.stats as bas
import pandas as pd
import plotly.graph_objects as go

from braian.ontology import AllenBrainOntology, UPPER_REGIONS

__all__ = [
    "plot_permutation",
    "plot_groups_salience",
    #"plot_latent_component",
    "plot_latent_variable",
    "plot_salient_regions",
]

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