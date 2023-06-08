import igraph as ig
import numpy as np
import plotly.graph_objects as go

from plotly.colors import DEFAULT_PLOTLY_COLORS
from .connectome import Connectome
from ..brain_hierarchy import AllenBrainHierarchy

def draw_network_plot(connectome: Connectome,
                      layout_fun: ig.Layout, AllenBrain: AllenBrainHierarchy,
                      use_centrality=False, centrality_metric=None, colorscale="Plasma",
                      **kwargs):
    graph_layout = layout_fun(connectome.G)
    nodes_trace = draw_nodes(connectome.G, graph_layout, 15, AllenBrain, outline_size=6,
                             use_centrality=use_centrality, centrality_metric=centrality_metric)
    if use_centrality and centrality_metric in connectome.G.vs.attributes():
        nodes_trace.marker.colorscale = colorscale
        nodes_trace.marker.showscale = True
        nodes_trace.marker.colorbar=dict(title=centrality_metric, len=0.5, thickness=15)
    edges_trace = draw_edges(connectome.G, graph_layout, 2)
    fig = go.Figure([edges_trace, nodes_trace],
                    layout=dict(
                        width=1000, height=1000,
                        xaxis=no_axis,
                        yaxis=no_axis,
                        paper_bgcolor='rgba(255,255,255,255)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        **kwargs
              ))
    return fig

def nodes_hover_info(G: ig.Graph, AllenBrain: AllenBrainHierarchy, title_dict: dict={}):
    customdata = []
    hovertemplates = []
    i = 0
    # Add vertices' attributes
    for attr in G.vs.attributes():
        match attr:
            case "name":
                customdata.extend((
                     G.vs["name"],
                     [AllenBrain.full_name[acronym] for acronym in G.vs["name"]]
                ))
                hovertemplates.extend((
                    f"Region: <b>%{{customdata[{i}]}}</b>",
                    f"<i>%{{customdata[{i+1}]}}</i>"
                ))
                i += 2
            case "upper_region":
                customdata.extend((
                     G.vs["upper_region"],
                     [AllenBrain.full_name[acronym] for acronym in G.vs["upper_region"]]
                ))
                hovertemplates.append(f"Major Division: %{{customdata[{i}]}} (%{{customdata[{i+1}]}})")
                i += 2
            case _:
                customdata.append(G.vs[attr])
                attr_title = title_dict[attr] if attr in title_dict else attr
                hovertemplates.append(f"{attr_title}: %{{customdata[{i}]}}")
                i += 1
    # Add additional information
    # fun is expected to be a function that takes a VertexSeq and spits a value for each vertex.
    for attr_title, fun in title_dict.items():
        if attr_title in G.vs.attributes():
            continue
        customdata.append(fun(G.vs))
        hovertemplates.append(f"{attr_title}: %{{customdata[{i}]}}")
        i += 1

    hovertemplate = "<br>".join(hovertemplates)
    hovertemplate += "<extra></extra>"
    # customdata=np.hstack((old_customdata.customdata, np.expand_dims(<new_data>, 1))), # update customdata
    return np.stack(customdata, axis=-1), hovertemplate

def draw_nodes(G: ig.Graph, graph_layout: ig.Layout, node_size: int, AllenBrain: AllenBrainHierarchy,
               outline_size=0.5, use_centrality=False, centrality_metric: str=None):
    colors = AllenBrain.get_region_colors()
    nodes_color = []
    outlines_color = []
    if "cluster" in G.vs.attributes():
        get_outline_color = lambda v: DEFAULT_PLOTLY_COLORS[v["cluster"] % len(DEFAULT_PLOTLY_COLORS)]
    else:
        get_outline_color = lambda v: colors[v["name"]]
    if use_centrality and (centrality_metric is None or centrality_metric not in G.vs.attributes()):
        raise ValueError("If you want to plot the centrality, you must also specify a nodes' attribute in 'centrality_metric'")
    for v in G.vs:
        if v.degree() > 0:
            outline_color = get_outline_color(v)
            node_color = v[centrality_metric] if use_centrality else colors[v["name"]]
        elif "is_undefined" in v.attributes() and v["is_undefined"]:
            outline_color = 'rgb(140,140,140)'
            node_color = '#A0A0A0'
        else:
            outline_color = 'rgb(150,150,150)'
            node_color = '#CCCCCC'
        nodes_color.append(node_color)
        outlines_color.append(outline_color)

    customdata, hovertemplate = nodes_hover_info(G, AllenBrain, title_dict={"Degree": ig.VertexSeq.degree})
    nodes_trace = go.Scatter(
        x=[coord[0] for coord in graph_layout.coords],
        y=[coord[1] for coord in graph_layout.coords],
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

def draw_edges(G: ig.Graph, layout: ig.Layout, width: int):
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
        mode="lines+markers" if G.is_directed() else "lines",
        showlegend=False)
    
    if G.is_directed():
        edges_trace.marker = dict(
                symbol="arrow",
                size=10,
                angleref="previous",
                standoff=8,
            )
    
    return edges_trace

no_axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=""
          )