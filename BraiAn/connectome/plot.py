import igraph as ig
import numpy as np
import plotly.graph_objects as go

from plotly.colors import DEFAULT_PLOTLY_COLORS
from .functional import FunctionalConnectome
from .utils_bu import participation_coefficient
from ..brain_hierarchy import AllenBrainHierarchy

def draw_network_plot(connectome: FunctionalConnectome,
                      layout_fun: ig.Layout, AllenBrain: AllenBrainHierarchy,
                      **kwargs):
    graph_layout = layout_fun(connectome.G)
    nodes_trace = draw_nodes(connectome.G, graph_layout, 15, AllenBrain)
    if connectome.vc is not None:
        add_participation_coefficient(nodes_trace, connectome)
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

def draw_nodes(G, layout, node_size, AllenBrain):
    colours = AllenBrain.get_region_colours()
    nodes_colour = []
    outlines_colour = []
    for v in G.vs:
        if v.degree() > 0:
            # node_colour = DEFAULT_PLOTLY_COLORS[v["cluster"]] if "cluster" in v.attribute_names() else colours[v["name"]]
            # outline_colour = node_colour # '#FFFFFF'
            # outline_colour = colours[v["name"]] # '#FFFFFF'
            outline_colour= DEFAULT_PLOTLY_COLORS[v["cluster"] % len(DEFAULT_PLOTLY_COLORS)]
            node_colour = colours[v["name"]]
            # node_colour = participations[v.index]
        elif v["is_undefined"]:
            outline_colour = 'rgb(140,140,140)'
            node_colour = '#A0A0A0'
        else:
            outline_colour = 'rgb(150,150,150)'
            node_colour = '#CCCCCC'
        nodes_colour.append(node_colour)
        outlines_colour.append(outline_colour)

    nodes_trace = go.Scatter(
        x=[coord[0] for coord in layout.coords],
        y=[coord[1] for coord in layout.coords],
        mode="markers",
        name="",
        marker=dict(symbol="circle",
                    size=node_size,
                    color=nodes_colour,
                    line=dict(color=outlines_colour, width=6)), #0.5)),
        customdata = np.stack((
                        G.vs["name"],
                        [AllenBrain.full_name[acronym] for acronym in G.vs["name"]],
                        G.vs["upper_region"],
                        [AllenBrain.full_name[acronym] for acronym in G.vs["upper_region"]],
                        G.vs.degree()),
                    axis=-1),
        hovertemplate=
            "Region: <b>%{customdata[0]}</b><br>" +
            "<i>%{customdata[1]}</i><br>" +
            "Major Division: %{customdata[2]} (%{customdata[3]})<br>" +
            "Degree: %{customdata[4]}" +
            "<extra></extra>",
        showlegend=False
    )

    return nodes_trace

def add_participation_coefficient(nodes_trace, connectome):
    participations = participation_coefficient(connectome.G, connectome.vc)
    nodes_trace.marker.color = list(participations)
    nodes_trace.marker.colorscale="Plasma"
    nodes_trace.marker.showscale = True
    nodes_trace.customdata = np.hstack((nodes_trace.customdata, np.expand_dims(participations, 1)))
    nodes_trace.hovertemplate = nodes_trace.hovertemplate + "<br>Participation coefficient: %{customdata[5]}"
    return

def draw_edges(G, layout, width):
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
        mode="lines",
        showlegend=False)
    
    return edges_trace

no_axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=""
          )