import plotly.graph_objects as go
import pandas as pd
import numpy as np
import igraph as ig

from .brain_hierarchy import AllenBrainHierarchy, UPPER_REGIONS
from .connectome.functional import FunctionalConnectome

def draw_chord_plot(connectome: FunctionalConnectome,
                    AllenBrain: AllenBrainHierarchy,
                    title="",
                    size=1500,
                    no_background=True,
                    regions_size=15,
                    regions_font_size=10,
                    max_edge_width=5,
                    use_weighted_edge_widths=True,
                    colorscale_edges=True,
                    colorscale="RdBu_r",
                    colorscale_min="cutoff",
                    ideograms_arc_index=50,
                    **kwargs):
    G = connectome.G

    circle_layout = G.layout_circle()
    circle_layout.rotate(180/len(circle_layout)) # rotate by half of a unit to sync with ideograms' rotation
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
              hovermode="closest",
              paper_bgcolor=paper_bgcolor,
              plot_bgcolor='rgba(0,0,0,0)',
              annotations=extract_annotations(kwargs, pos=-0.07, step=-0.02)
              )

    nodes = draw_nodes(layout, G, circle_layout, regions_size, regions_font_size, colours, AllenBrain)
    lines, edge_info, colorbar = draw_edges(layout, G, circle_layout, connectome.r_cutoff, max_edge_width, use_weighted_edge_widths,
                                            use_colorscale=colorscale_edges, colorscale=colorscale,
                                            colorscale_min=connectome.r_cutoff if colorscale_min == "cutoff" else colorscale_min)
    ideograms = draw_ideograms(layout, G.vs["upper_region"], AllenBrain, colours, a=ideograms_arc_index)

    fig = go.Figure(data=ideograms+lines+[nodes]+edge_info+colorbar, layout=layout)
    return fig

def draw_nodes(layout, G, circle_layout, node_size, font_size, colours, AllenBrain):
    nodes_colour = []
    outlines_colour = []
    for v in G.vs:
        if v.degree() > 0:
            outline_colour = '#FFFFFF'
            node_colour = colours[v["label"]]
        elif v["is_undefined"]:
            outline_colour = 'rgb(140,140,140)'
            node_colour = '#A0A0A0'
        else:
            outline_colour = 'rgb(150,150,150)'
            node_colour = '#CCCCCC'
        nodes_colour.append(node_colour)
        outlines_colour.append(outline_colour)

    nodes = go.Scatter(
        x=[coord[0] for coord in circle_layout.coords],
        y=[coord[1] for coord in circle_layout.coords],
        mode="markers",
        name="",
        marker=dict(symbol="circle",
                    size=node_size,
                    color=nodes_colour,
                    line=dict(color=outlines_colour, width=0.5)),
        customdata = np.stack((
                        G.vs["label"],
                        [AllenBrain.full_name[acronym] for acronym in G.vs["label"]],
                        G.vs["upper_region"],
                        [AllenBrain.full_name[acronym] for acronym in G.vs["upper_region"]],
                        G.vs.degree()),
                    axis=-1),
        hovertemplate=
            "Region: <b>%{customdata[0]}</b><br>" +
            "<i>%{customdata[1]}</i><br>" +
            "Major Division: %{customdata[2]} (%{customdata[3]})<br>" +
            "Degree: %{customdata[4]}" +
            "<extra></extra>"
        )
    nodes_annotations = []
    for v in G.vs:
        x = circle_layout[v.index][0]
        y = circle_layout[v.index][1]
        angle = 180/PI*np.arctan(x/y)-90 if y != 0 else 0
        xanchor = "right" if x < 0 else "left"
        yanchor = "bottom" if y < 0 else "top"
        if abs(angle) > 90:
            angle = ((angle+90)%180)-90
        textdist = 20
        ann = dict(
                x=x,
                y=y,
                text=v["label"],
                xshift=x*textdist,
                yshift=y*textdist,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                textangle=angle,
                font_size=font_size,
                showarrow=False
            )
        nodes_annotations.append(ann)
    layout["annotations"] = (*layout["annotations"], *nodes_annotations)

    return nodes

def draw_edges(layout, G: ig.Graph, circle_layout, r_cutoff, max_width, use_weighted_widths,
               solid_color="#5588c8", use_colorscale=True, colorscale="RdBu_r", colorscale_min=0):
    if use_colorscale:
        colorbar_trace = go.Scatter(x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale, 
                    showscale=True,
                    cmin=colorscale_min,
                    cmax=1,
                    colorbar=dict(title="Pearson's <i>r</i>", len=0.5, thickness=15), 
                ),
                hoverinfo='none'
                )

    lines = [] # the list of dicts defining   edge  Plotly attributes
    edge_info = [] # the list of points on edges where  the information is placed
    if len(G.es) == 0:
        return lines, edge_info, [colorbar_trace if use_colorscale else None]

    Dist = [0, dist([1,0], 2*[np.sqrt(2)/2]), np.sqrt(2), dist([1,0],  [-np.sqrt(2)/2, np.sqrt(2)/2]), 2.0]
    params = [1.2, 1.5, 1.8, 2.1]
    sorted_es = sorted(G.es, key=lambda e: e["weight"] if G.is_weighted() else e["r-value"])
    sorted_rvalues = [e["weight"] if G.is_weighted() else e["r-value"] for e in sorted_es]
    if use_weighted_widths:
        edges_widths = get_edges_widths(sorted_rvalues, r_cutoff, max=max_width) #The width is proportional to Pearson's r value
    if use_colorscale:
        edge_colours=[get_color(colorscale, c, min_loc=colorscale_min, max_loc=1) for c in sorted_rvalues]

    for j, e in enumerate(sorted_es):
        A=np.array(circle_layout[e.source])
        B=np.array(circle_layout[e.target])
        d=dist(A, B)
        K=get_idx_interv(d, Dist)
        b=[A, A/params[K], B/params[K], B]
        pts=BezierCv(b, nr=5)
        text="<br>".join((f"<b>{e.source_vertex['label']} - {e.target_vertex['label']}</b>",
                        f"r: {e['weight'] if G.is_weighted() else e['r-value']:.5f}",
                        f"p: {e['p-value']:.10f}"))
        mark1=deCasteljau(b,0.9)
        mark2=deCasteljau(list(reversed(b)),0.9)
        edge_info.append(go.Scatter(x=[mark1[0], mark2[0]],
                                y=[mark1[1], mark2[1]],
                                mode="markers",
                                marker=dict(size=0.5, color=edge_colours[j] if use_colorscale else solid_color),
                                text=text,
                                hoverinfo="text"
                                )
                        )
        lines.append(go.Scatter(x=pts[:,0],
                            y=pts[:,1],
                            mode="lines",
                            line=dict(
                                shape="spline",
                                width=edges_widths[j] if use_weighted_widths else max_width,
                                color=edge_colours[j] if use_colorscale else solid_color,
                            ),
                            hoverinfo="none"
                        )
                    )
    return lines, edge_info, [colorbar_trace if use_colorscale else None]

def draw_ideograms(layout, all_upper_regions, AllenBrain,
                    colours, outline_colour='rgb(150,150,150)',
                    a=50):
    upper_regions = sorted(list(set(all_upper_regions)), key=UPPER_REGIONS.index)
    n_regions_in_upper = np.asarray([sum(r1 == r2  for r2 in all_upper_regions) for r1 in upper_regions])
    ideograms=[]

    n_nodes = n_regions_in_upper.sum()
    ideogram_length=2*PI*n_regions_in_upper/n_nodes
    ideo_ends = get_ideogram_ends(ideogram_length)
    ideo_colours = [colours[r_acronym] for r_acronym in upper_regions]

    for k in range(len(ideo_ends)):
        z= make_ideogram_arc(1.2, ideo_ends[k], a=a)
        zi=make_ideogram_arc(1.0, ideo_ends[k], a=a)
        m=len(z)
        n=len(zi)
        ideograms.append(go.Scatter(x=z.real,
                                y=z.imag,
                                mode='lines',
                                line=dict(color=ideo_colours[k], shape='spline', width=0.25),
                                text=f"<b>{upper_regions[k]}</b><br>"+
                                        f"<i>{AllenBrain.full_name[upper_regions[k]]}</i><br>"
                                        f"N displayed regions: {n_regions_in_upper[k]:d}",
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

def get_color(colorscale_name, loc, min_loc=0, max_loc=1):
    if not(min_loc < loc < max_loc):
        raise ValueError(f"'min_loc' ({min_loc}) < 'loc' ({loc}) < 'max_loc' ({max_loc}) inequality is not respected.")
    if min != 0 or max != 1:
        loc = (loc-min_loc) / (max_loc-min_loc)
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
PI = np.pi

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

def extract_annotations(kwargs, pos=-0.07, step=-0.02):
    annotations = []
    for key in kwargs:
        if key.startswith("annotation"):
            bottom_annotation = make_annotation(kwargs[key], pos)
            annotations.append(bottom_annotation)
            pos += step
        elif key == "subtitle":
            subtitle = dict(showarrow=False,
                    text=kwargs[key],
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=15))
            annotations.append(subtitle)
    return annotations

def make_annotation(anno_text, y_coord, x_coord=0, font_size=12):
    return dict(showarrow=False,
                text=anno_text,
                xref='paper',
                yref='paper',
                x=x_coord,
                y=y_coord,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=font_size)
            )