from .brain_hierarchy import *
from .pls_helpers import *
from .brain_slice import *
from .sliced_brain import *
from .animal_brain import *
from .animal_group import *

def plot_cv_above_threshold(brains_CV, brains_name, marker, cv_threshold=1) -> go.Figure: 
    fig = go.Figure()
    for i,cv in enumerate(brains_CV):
        above_threshold_filter = cv > cv_threshold
        # Scatterplot (animals)
        fig.add_trace(go.Scatter(
                            mode = 'markers',
                            y = cv[above_threshold_filter],
                            x = [i]*above_threshold_filter.sum(),
                            text = cv.index[above_threshold_filter],
                            opacity=0.7,
                            marker=dict(
                                size=7,
                                line=dict(
                                    color='rgb(0,0,0)',
                                    width=1
                                )
                            ),
                            showlegend=False
                    )
        )

    fig.update_layout(
        title = f"Coefficient of variaton of {marker} across brain slices > {cv_threshold}",
        
        xaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(0,len(brains_name)),
            ticktext = brains_name
        ),
        yaxis=dict(
            title = "Brain regions' CV"
        ),
        width=700, height=500
    )
    return fig