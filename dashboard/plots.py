"""
Responsibilities:
1. Accuracy vs Carbon Emissions scatter plot
2. GreenScore leaderboard bar chart

NOTE:
- Uses Plotly for interactive visuals
- UI layout handled in app.py
"""

import plotly.express as px


# -------------------------------------------------
# Accuracy vs Carbon Emissions
# -------------------------------------------------
def plot_accuracy_vs_carbon(df):
    """
    Creates a scatter plot showing the trade-off
    between model accuracy and carbon emissions.

    X-axis  : Accuracy
    Y-axis  : CO2 emissions (kg)
    Bubble  : GreenScore
    Color   : Model name

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe

    Returns
    -------
    plotly.graph_objects.Figure
    """

    fig = px.scatter(
        df,
        x="Accuracy",
        y="CO2 (kg)",
        color="Model",
        size="GreenScore",
        hover_data=["Energy (kWh)", "Time (s)"],
        template="plotly_dark",
        title="Accuracy vs Carbon Emissions"
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_title="Accuracy",
        yaxis_title="Carbon Emissions (kg)",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


# -------------------------------------------------
# GreenScore Leaderboard
# -------------------------------------------------
def plot_greenscore_bar(df):
    """
    Creates a bar chart ranking models by GreenScore.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe

    Returns
    -------
    plotly.graph_objects.Figure
    """

    sorted_df = df.sort_values("GreenScore", ascending=False)

    fig = px.bar(
        sorted_df,
        x="Model",
        y="GreenScore",
        color="Model",
        text="GreenScore",
        template="plotly_dark",
        title="GreenScore Leaderboard"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        title_x=0.5,
        showlegend=False,
        yaxis_range=[0, 1]
    )

    return fig
