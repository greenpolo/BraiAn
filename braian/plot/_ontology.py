import igraph as ig
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go

from collections.abc import Iterable, Callable
from typing import Literal

from braian import _graph_utils, AtlasOntology, BrainData

__all__ = [
    "hierarchy",
    "draw_edges",
    "draw_nodes"
]

def hierarchy(brain_ontology: AtlasOntology,
              bdata: BrainData=None,
              selection: bool=True,
              unreferenced: bool=False,
              blacklisted: bool=True) -> go.Figure:
    """
    Plots the ontology as a tree.

    Parameters
    ----------
    brain_ontology
        The instance of a brain ontology to plot.
    bdata
        If provided, it projects some brain data onto the ontology tree.
    selection
        If True, it plots an horizontal red line that cuts through all currently selected regions in the ontology
    unreferenced
        If False, it hides all those brain regions that have no references in the atlas annotations.
    blacklisted
        If False, it hides all those brain regions that are currently blacklisted from the ontology.\
        If `unreferenced` is True, it is ignored.

    Returns
    -------
    :
        A plotly Figure
    """
    g: ig.Graph = brain_ontology.to_igraph(unreferenced=unreferenced, blacklisted=blacklisted)
    graph_layout = g.layout_reingold_tilford(mode="out", root=[0])
    edges_trace = draw_edges(g, graph_layout, width=0.5, directed=False)
    nodes_params = dict(
        layout=graph_layout,
        brain_ontology=brain_ontology,
        node_size=5,
        metrics={"Subregions": lambda vs: np.asarray(vs.degree())-1}
    )
    if bdata is not None:
        selected_regions = set(bdata.regions) - set(bdata.missing_regions())
        data = bdata.data.astype(float) # if bdata had pd.NA (missing data), plotly would fail
        for v in g.vs:
            acronym = v["name"]
            v["cluster"] = 2 if acronym in selected_regions else 1
            v[f"{bdata.units} - {bdata.metric}"] = data[acronym] if acronym in data else np.nan
        # nodes_params |= dict(outline_mode="cluster", outline_size=0.5)
        nodes_params |= dict(fill_mode="cluster", cluster_colors=["#EF553B", "#00CC96"]) # not in bdata, in bdata
    nodes_trace = draw_nodes(g, **nodes_params)
    nodes_trace.marker.line = dict(color="black", width=0.25)
    traces = [edges_trace, nodes_trace]
    if selection and brain_ontology.has_selection():
        traces.append(draw_selection(g, graph_layout, width=2))
    plot_layout = go.Layout(
        title="Allen's brain region hierarchy",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, dtick=1, autorange="reversed", title="depth"),
        template="none"
    )
    return go.Figure(traces, layout=plot_layout)

def draw_edges(G: ig.Graph, layout: ig.Layout, width: int, directed: bool=True) -> go.Scatter:
    """
    Draws a plotly Line plot of the given graph `G`, based on the given layout.

    Parameters
    ----------
    G
        A graph.
    layout
        The layout used to position the nodes of the graph `G`.
    width
        The width of the edges' lines.
    directed
        If True and `G` is a directed graph, it draws edges as arrows.

    Returns
    -------
    :
        A plotly scatter trace
    """
    edge_x = []
    edge_y = []
    for e in G.es:
        x0, y0 = layout.coords[e.source]
        x1, y1 = layout.coords[e.target]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=width, color="#888"),
        hoverinfo="none",
        mode="lines+markers" if directed and G.is_directed() else "lines",
        showlegend=False)

    if directed and G.is_directed():
        edges_trace.marker = dict(
                symbol="arrow",
                size=10,
                angleref="previous",
                standoff=8,
            )

    return edges_trace

def draw_nodes(G: ig.Graph, layout: ig.Layout, brain_ontology: AtlasOntology,
               node_size: int,
               fill_mode: Literal["region","centrality", "cluster"]="region",
               outline_size: float=0.5, outline_mode: Literal["region","centrality", "cluster"]="region",
               centrality_metric: str=None,
               metrics: dict[str,Callable[[ig.VertexSeq],Iterable[float]]]={"degree": ig.VertexSeq.degree},
               cluster_colors: list[str]=plc.qualitative.Plotly
               ) -> go.Scatter:
    """
    Draws a plotly Scatter plot of the given graph `G`, based on the given layout.

    Parameters
    ----------
    G
        A graph
    layout
        The layout used to position the nodes of the graph `G`
    brain_ontology
        The atlas ontology associated with `G`. It is used to retrieve information about brain regions.
    node_size
        The size of the region nodes
    fill_mode
        The mode chosen to color each region node.
        Accepted values are: `"region"`, `"centrality"` or `"cluster"`.
    outline_size
        The size of the region nodes' outlines
    outline_mode
        The mode chosen to color each region node outline.
        Accepted values are: `"region"`, `"centrality"` or `"cluster"`.
        If `"region"`, it colors based on `brain_ontology`.
        If `"centrality"`, it colors based on the attribute defined in `centrality_metric` of `G` vertices.
        If `"cluster"`, it colors based on `cluster` attribute on `G` vertices.
    centrality_metric
        The name of the attribute used if `fill_mode` or `outline_mode` are set to `"centrality"`.
    metrics
        A dictionary that defines M additional information for the vertices of graph `G`.
        The keys are title of an additional metric, while the values are functions that
        take a `igraph.VertexSeq` and spits a value for each vertex.

    Returns
    -------
    :
        A plotly scatter trace

    Raises
    ------
    ValueError
        If `fill_mode` or `outline_mode` are set to `"centrality"`,
        but `G` vertices have no attributes named `"cluster"`.
    ValueError
        If `fill_mode` or `outline_mode` are set to `"centrality"`,
        but `G` vertices have no attributes named as defined by `centrality_metric`.
    """
    region_colors = brain_ontology.colors()
    vertex_fill =       _region_color(fill_mode, G, region_colors, centrality_metric, cluster_colors=cluster_colors)
    vertex_outline = _region_color(outline_mode, G, region_colors, centrality_metric, cluster_colors=cluster_colors)
    nodes_color = []
    outlines_color = []

    for v in G.vs:
        if v.degree() > 0:
            outline_color = vertex_outline(v)
            node_color = vertex_fill(v)
        elif "is_undefined" in v.attributes() and v["is_undefined"]:
            outline_color = "rgb(140,140,140)"
            node_color = "rgb(160,160,160)"
        else:
            outline_color = "rgb(150,150,150)"
            node_color = "rgb(204,204,204)"
        nodes_color.append(node_color)
        outlines_color.append(outline_color)

    customdata, hovertemplate = _nodes_hover_info(brain_ontology, G, title_dict=metrics)
    nodes_trace = go.Scatter(
        x=[coord[0] for coord in layout.coords],
        y=[coord[1] for coord in layout.coords],
        mode="markers",
        name="",
        marker=dict(symbol="circle",
                    size=node_size,
                    color=nodes_color,
                    line=dict(color=outlines_color, width=outline_size)),
        customdata=customdata,
        hovertemplate=hovertemplate,
        showlegend=False
    )
    return nodes_trace

def draw_selection(g: ig.Graph, layout: ig.Layout, width: float):
    coords = []
    contiguous_selections = _graph_utils.brainwide_selection(g, attr="index")
    for contiguous_selection in contiguous_selections:
        for v in contiguous_selection:
            coords.append(layout.coords[v])
        coords.append([None,None])
    coords = np.array(coords)
    return go.Scatter(
        x=coords[:,0], y=coords[:,1],
        line=dict(width=width, color="red"),
        hoverinfo="none",
        mode="lines",
        showlegend=True)

def _region_color(mode: Literal["region","centrality", "cluster"],
                  G: ig.Graph,
                  region_colors: list[str],
                  centrality_metric: str,
                  cluster_colors: list[str]=plc.qualitative.Plotly) -> Callable[[ig.Vertex],str]:
    match mode:
        case "region":
            return lambda v: region_colors[v["name"]]  # noqa: E731
        case "centrality":
            if centrality_metric in G.vs.attributes():
                return lambda v: v[centrality_metric]
            elif centrality_metric in (dir:=G.vs.__dir__()):
                # return lambda v: dir[centrality_metric](v)
                return lambda v: v.__getattribute__(centrality_metric)()
            else:
                raise ValueError("No valid centrality metric was specified.") # Check the vertices' attributes to see the available options.")
        case "cluster":
            if "cluster" not in G.vs.attributes():
                raise ValueError("No clustering was made on the provided connectome")
            cc = pd.Categorical(G.vs["cluster"])
            return lambda v: cluster_colors[cc.codes[v.index] % len(cluster_colors)]  # noqa: E731

def _nodes_hover_info(brain_ontology: AtlasOntology, G: ig.Graph,
                     title_dict: dict[str,Callable[[ig.VertexSeq],Iterable[float]]]={}
                     ) -> tuple[npt.NDArray,str]:
    """
    Computes the information when hovering over a vertex of the graph `G`.
    It allows to add additional information based on given functions.
    Returns a tuple where the first element is an matrix of custom data,
    while the second element is a hover template.
    Both are used by plotly to modify information when hovering points in a Scatter plot

    Parameters
    ----------
    brain_ontology
        TODO: missing
    G
        A graph with N vertices
    title_dict
        A dictionary that defines M additional information for the vertices of graph `G`.
        The keys are title of an additional metric, while the values are functions that
        take a `igraph.VertexSeq` and spits a value for each vertex.

    Returns
    -------
    :
        A customdata M×N matrix and a hovertemplate
    """
    customdata = []
    hovertemplates = []
    i = 0
    # Add vertices' attributes
    for attr in G.vs.attributes():
        match attr:
            case "name":
                customdata.extend((
                    G.vs["name"],
                    [brain_ontology.full_name[acronym] for acronym in G.vs["name"]]
                ))
                hovertemplates.extend((
                    f"Region: <b>%{{customdata[{i}]}}</b>",
                    f"<i>%{{customdata[{i+1}]}}</i>"
                ))
                i += 2
            case "upper_region":
                customdata.extend((
                    G.vs["upper_region"],
                    [brain_ontology.full_name[acronym] for acronym in G.vs["upper_region"]]
                ))
                hovertemplates.append(f"Major Division: %{{customdata[{i}]}} (%{{customdata[{i+1}]}})")
                i += 2
            case _:
                if attr.lower() in (t.lower() for t in title_dict.keys()):
                    # If one of the additional information wants to overload an attribute (e.g. 'depth')
                    # then skip it
                    continue
                customdata.append(G.vs[attr])
                attribute_title = attr.replace("_", " ").title()
                hovertemplates.append(f"{attribute_title}: %{{customdata[{i}]}}")
                i += 1
    # Add additional information
    for custom_title, fun in title_dict.items():
        customdata.append(fun(G.vs))
        hovertemplates.append(f"{custom_title.title()}: %{{customdata[{i}]}}")
        i += 1

    hovertemplate = "<br>".join(hovertemplates)
    hovertemplate += "<extra></extra>"
    # customdata=np.hstack((old_customdata.customdata, np.expand_dims(<new_data>, 1))), # update customdata
    return np.stack(customdata, axis=-1), hovertemplate