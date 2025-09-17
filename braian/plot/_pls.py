import braian.stats as bas
import plotly.graph_objects as go

__all__ = [
    "permutation",
    "groups_salience",
    #"plot_latent_component",
    "latent_variable",
]

def permutation(pls: bas.PLS, component: int=1) -> go.Figure:
    """
    Plots the result of [`PLS.random_permutation()`][braian.stats.PLS.random_permutation].
    It shows how much the product of the given partial least square analysis is a result of pure chance.

    Parameters
    ----------
    pls
        An instance of a mean-centered task partial least square analysis.
    component
        The component of the PLS for which to plot the permutation.

    Returns
    -------
    :
        A Plotly figure.
    
    See also
    --------
    [`PLS.n_components`][braian.stats.PLS.n_components]
    """
    n,_ = pls.s_sampling_distribution.shape
    assert 1 <= component < pls.n_components(), f"'component' must be between 1 and {pls.n_components()}."
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

def groups_salience(pls: bas.PLS, component: int=1) -> go.Figure:
    """
    Bar plot of the salience scores of each group in the [`PLS`][braian.stats.PLS].

    Parameters
    ----------
    pls
        An instance of a mean-centered task partial least square analysis.
    component
        The component of the PLS for which to plot the permutation.

    Returns
    -------
    :
        A Plotly figure.
    
    See also
    --------
    [`PLS.n_components`][braian.stats.PLS.n_components]
    """
    assert 1 <= component < pls.n_components(), f"'component' must be between 1 and {pls.n_components()}."
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

def latent_variable(pls: bas.PLS, of: str="X", width: int=800, height: int=800) -> go.Figure:
    """
    PCA-like plot of [_brain scores_][braian.stats.PLS.Lx] or [_group scores_][braian.stats.PLS.Ly] of a PLS.
    This might help see how animals or groups are discernable from each other.
    
    It always plots the first component on the x-axis and the second component on the y-axis.

    Parameters
    ----------
    pls
        An instance of a mean-centered task partial least square analysis.
    of
        If "X", it plots the [_brain scores_][braian.stats.PLS.Lx].
        If "Y", it plots the [_group scores_][braian.stats.PLS.Ly].
    width
        The width of the plot.
    height
        The height of the plot.

    Returns
    -------
    :
        A Plotly figure.
    """
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