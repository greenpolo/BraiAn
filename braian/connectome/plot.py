import igraph as ig
import plotly.graph_objects as go

from braian.connectome.connectome import Connectome
from braian.brain_hierarchy import AllenBrainHierarchy

def draw_network_plot(connectome: Connectome,
                      layout_fun: ig.Layout, brain_ontology: AllenBrainHierarchy,
                      use_centrality=False, centrality_metric=None, colorscale="Plasma",
                      use_clustering=False, isolated_regions=False,
                      width=None, height=1000,
                      **kwargs):
    if not isolated_regions:
        connected_vs = connectome.G.vs.select(_degree_ge=1)
        G = connectome.G.induced_subgraph(connected_vs)
    else:
        G = connectome.G
    graph_layout = layout_fun(G)
    nodes_trace = brain_ontology.draw_nodes(G, graph_layout, 15, outline_size=6,
                             use_centrality=use_centrality, centrality_metric=centrality_metric,
                             use_clustering=use_clustering)
    if use_centrality and centrality_metric in G.vs.attributes():
        nodes_trace.marker.colorscale = colorscale
        nodes_trace.marker.showscale = True
        nodes_trace.marker.colorbar=dict(title=centrality_metric, len=0.5, thickness=15)
    edges_trace = brain_ontology.draw_edges(G, graph_layout, 2)
    fig = go.Figure([edges_trace, nodes_trace],
                    layout=dict(
                        width=width, height=height,
                        xaxis=no_axis,
                        yaxis=no_axis,
                        paper_bgcolor='rgba(255,255,255,255)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        **kwargs
              ))
    return fig

no_axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=""
          )