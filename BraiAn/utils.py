import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def save_csv(df: pd.DataFrame, output_path: str, file_name:str, overwrite=False, sep="\t") -> None:
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)
    if os.path.exists(file_path):
        if not overwrite:
            raise FileExistsError(f"The file {file_name} already exists in {output_path}!")
        else:
            print(f"WARNING: The file {file_name} already exists in {output_path}. Overwriting previous CSV!")
    df.to_csv(file_path, sep=sep, mode="w")

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