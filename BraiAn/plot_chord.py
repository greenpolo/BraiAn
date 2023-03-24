import plotly.graph_objects as go
import pandas as pd
import numpy as np
import igraph as ig

from .brain_hierarchy import MAJOR_DIVISIONS, AllenBrainHierarchy


def draw_chord_plot(r: pd.DataFrame, p: pd.DataFrame, r_cutoff, p_cutoff,
                    AllenBrain: AllenBrainHierarchy,
                    ideograms_a=50,
                    title="",
                    size=1500,
                    no_background=True,
                    colorscale_edges=True,
                    **kwargs):
    above_threshold = (p.abs() <= p_cutoff) & (r.abs() >= r_cutoff)
#    above_threshold = (p.abs() <= p_cutoff) & (r <= -r_cutoff)
    A = r.copy(deep=True)
    A[~above_threshold] = 0
    regions_MD = AllenBrain.get_areas_major_division(*A.index)

    # sort vertices based on major divisions
    regions_i = sorted(range(len(A)), key=lambda i: MAJOR_DIVISIONS.index(list(regions_MD.values())[i]))

    active_MD = sorted(list(set(regions_MD.values())), key=MAJOR_DIVISIONS.index)
    n_MD = [sum(r1 == r2  for r2 in regions_MD.values()) for r1 in active_MD]

    A = A.iloc[regions_i].iloc[:,regions_i]
    G = ig.Graph.Weighted_Adjacency(A.values, mode="lower") # mode="undirected")
    G.vs["label"] = list(A.index)
    G.vs["major_division"] = [regions_MD[v["label"]] for v in G.vs]
    circle_layout = G.layout_circle()
    colours = AllenBrain.get_region_colours()

    paper_bgcolor = 'rgba(0,0,0,0)' if no_background else 'rgba(255,255,255,255)'

    layout=go.Layout(
              title=title,
              font=dict(size=12),
              showlegend=False,
              autosize=False,
              width=size,
              height=size,
              xaxis=dict(axis),
              yaxis=dict(axis,
                        scaleanchor="x", scaleratio=1),
              margin=dict(l=40,
                            r=40,
                            b=185,
                            t=100,
                          ),
              hovermode='closest',
              paper_bgcolor=paper_bgcolor,
              plot_bgcolor='rgba(0,0,0,0)',
              annotations=extract_annotations(kwargs, pos=-0.07, step=-0.02)
              )

    nodes = draw_nodes(G, circle_layout, colours)
    lines, edge_info = draw_edges(G, circle_layout, r_cutoff, colorscale_edges)
    ideograms = draw_ideograms(layout, active_MD, n_MD, colours, a=ideograms_a)

    fig = go.Figure(data=ideograms+lines+[nodes]+edge_info, layout=layout)
    return fig

def draw_nodes(G, circle_layout, colours):
    node_colour = [colours[v["label"]] if v.degree() > 0 else '#CCCCCC' for v in G.vs]
    line_colour = ['#FFFFFF' if v.degree() > 0 else 'rgb(150,150,150)' for v in G.vs]
    region_labels = [f"Region: <b>{v['label']}</b><br>"+\
                    f"Major Division: {v['major_division']}<br>"+\
                    f"Degree: {v.degree()}"
                    for v in G.vs]

    nodes = go.Scatter(x=[coord[0] for coord in circle_layout.coords],
           y=[coord[1] for coord in circle_layout.coords],
           mode='markers',
           name='',
           marker=dict(symbol='circle',
                         size=15,
                         color=node_colour,
                         line=dict(color=line_colour, width=0.5)
                         ),
           text=region_labels,
           hoverinfo='text',
           )
    return nodes

def draw_edges(G, circle_layout, r_cutoff, use_colorscale):
    lines = [] # the list of dicts defining   edge  Plotly attributes
    edge_info = [] # the list of points on edges where  the information is placed
    
    Dist = [0, dist([1,0], 2*[np.sqrt(2)/2]), np.sqrt(2), dist([1,0],  [-np.sqrt(2)/2, np.sqrt(2)/2]), 2.0]
    params = [1.2, 1.5, 1.8, 2.1]
    edges_widths = get_edges_widths(G.es["weight"], r_cutoff) #The width is proportional to Pearson's r value
    if use_colorscale:
        edge_colours=[get_color("RdBu_r", (c+1)/2) for c in G.es['weight']]

    for j, e in enumerate(G.es):
        A=np.array(circle_layout[e.source])
        B=np.array(circle_layout[e.target])
        d=dist(A, B)
        K=get_idx_interv(d, Dist)
        b=[A, A/params[K], B/params[K], B]
#        colour=edge_colours[K]
        pts=BezierCv(b, nr=5)
        text=f"<b>{e.source_vertex['label']} - {e.target_vertex['label']}</b><br>r: {e['weight']}"
        mark=deCasteljau(b,0.9)
        edge_info.append(go.Scatter(x=[mark[0]],
                                y=[mark[1]],
                                mode='markers',
                                marker=dict(size=0.5, color=edge_colours),
                                text=text,
                                hoverinfo='text'
                                )
                        )
        lines.append(go.Scatter(x=pts[:,0],
                            y=pts[:,1],
                            mode='lines',
                            line=dict(
                                shape='spline',
                                width=edges_widths[j],
                                color=edge_colours[j] if use_colorscale else "#5588c8",
                            ),
                            hoverinfo='none'
                        )
                    )
    return lines, edge_info

# TODO: change names
def draw_ideograms(layout, active_MD, n_MD,
                    colours,outline_colour='rgb(150,150,150)',
                    a=50):
    assert len(active_MD) == len(n_MD), "The number of subregions per major division must be the same as the number of passed major divisions"
    ideograms=[]

    n_nodes = sum(n_MD)
    ideogram_length=2*PI*np.asarray(n_MD)/n_nodes
    ideo_ends = get_ideogram_ends(ideogram_length)-PI/n_nodes
    ideo_colours = [colours[r_acronym] for r_acronym in active_MD]

    for k in range(len(ideo_ends)):
        z= make_ideogram_arc(1.1, ideo_ends[k], a=a)
        zi=make_ideogram_arc(1.0, ideo_ends[k], a=a)
        m=len(z)
        n=len(zi)
        ideograms.append(go.Scatter(x=z.real,
                                y=z.imag,
                                mode='lines',
                                line=dict(color=ideo_colours[k], shape='spline', width=0.25),
                                text=f"{active_MD[k]}<br>{n_MD[k]:d}",
                                hoverinfo='text'
                                )
                        )


        path='M '
        for s in range(m):
            path+=str(z.real[s])+', '+str(z.imag[s])+' L '

        Zi=np.array(zi.tolist()[::-1])

        for s in range(m):
            path+=str(Zi.real[s])+', '+str(Zi.imag[s])+' L '
        path+=str(z.real[0])+' ,'+str(z.imag[0])

        layout.shapes = [*layout.shapes, make_ideo_shape(path, outline_colour, ideo_colours[k])]

    return ideograms
    


#############
# CHORD UTILS

def dist(A,B):
    return np.linalg.norm(np.array(A)-np.array(B))

def get_idx_interv(d, D):
    k=0
    while(d>D[k]):
        k+=1
    return  k-1

class InvalidInputError(Exception):
    pass

def deCasteljau(b,t):
    N=len(b)
    if(N<2):
        raise InvalidInputError("The  control polygon must have at least two points")
    a=np.copy(b) #shallow copy of the list of control points 
    for r in range(1,N):
        a[:N-r,:]=(1-t)*a[:N-r,:]+t*a[1:N-r+1,:]
    return a[0,:]

def BezierCv(b, nr=5):
    t=np.linspace(0, 1, nr)
    return np.array([deCasteljau(b, t[k]) for k in range(nr)])

def get_edges_widths(r_values, r_cutoff, max=5):
    return (np.abs(np.array(r_values))-r_cutoff)/(1-r_cutoff)*max

def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)

import plotly.colors

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

#################
# IDEOGRAMS UTILS
PI=np.pi

def moduloAB(x, a, b): #maps a real number onto the unit circle identified with 
                       #the interval [a,b), b-a=2*PI
        if a>=b:
            raise ValueError('Incorrect interval ends')
        y=(x-a)%(b-a)
        return y+b if y<0 else y+a

def test_2PI(x):
    return 0<= x <2*PI

def get_ideogram_ends(ideogram_len):
    ideo_ends=[]
    left=0
    for k in range(len(ideogram_len)):
        right=left+ideogram_len[k]
        ideo_ends.append([left, right])
        left=right
    return np.array(ideo_ends)


def make_ideogram_arc(R, phi, a=50):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
#    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
#        phi=[moduloAB(t, 0, 2*PI) for t in phi]
    length=(phi[1]-phi[0])% 2*PI
    nr=5 if length<=PI/4 else int(a*length/PI)

    if phi[0] < phi[1]:
        theta=np.linspace(phi[0], phi[1], nr)
    else:
        phi=[moduloAB(t, -PI, PI) for t in phi]
        theta=np.linspace(phi[0], phi[1], nr)
    return R*np.exp(1j*theta)

def make_ideo_shape(path, line_color, fill_color):
    #line_color is the color of the shape boundary
    #fill_collor is the color assigned to an ideogram
    return  dict(
                  line=dict(
                  color=line_color,
                  width=0.45
                 ),

            path=path,
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

############
# PLOT UTILS

axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

def extract_annotations(dict, pos=-0.07, step=-0.02):
    annotations = []
    for key in dict:
        if not key.startswith("annotation"):
            continue
        annotations.append(make_annotation(dict[key], pos))
        pos += step
    return annotations

def make_annotation(anno_text, y_coord):
    return dict(showarrow=False,
                      text=anno_text,
                      xref='paper',
                      yref='paper',
                      x=0,
                      y=y_coord,
                      xanchor='left',
                      yanchor='bottom',
                      font=dict(size=12)
                     )