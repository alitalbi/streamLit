import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def plot_assets_final_layout(asset_data, highlight_critical=False):


    fig = make_subplots(
        rows=len(asset_data)*2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.8,
        subplot_titles= [asset + " "+elem for asset in asset_data for elem in ["Fractal","Price"]]
    )

    i = 1
    for asset_name, df in asset_data.items():
        # ➔ This is the real calculation now
        # --- Fractal ---
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Fractal'],
                mode='lines+markers' if highlight_critical else 'lines',
                marker=dict(color=['red' if f <= 1.3 else 'blue' for f in df['Fractal']],
                            size=4) if highlight_critical else None,
                line=dict(color='blue') if not highlight_critical else None,
                name=f"{asset_name} Fractal"
            ),
            row=i, col=1
        )

        # --- Threshold 1.25 ---
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[1.25] * len(df),
                mode='lines',
                name="Critical 1.25",
                line=dict(color='red', dash='dash')
            ),
            row=i, col=1
        )

        # --- Price ---
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'].iloc[:,0],
                mode='lines',
                name=f"{asset_name} Price",
                marker=dict(color='white')
            ),
            row=i+1, col=1
        )

        i += 2

    fig.update_layout(
        autosize=True,
        yaxis_autorange=True,
        height=1600,
        width=800,
        showlegend=False,
        template="plotly_white",
        yaxis2_autorange=True,
        dragmode='zoom'
       # margin=dict(t=40, b=30, l=30, r=20)
    )

    return fig
