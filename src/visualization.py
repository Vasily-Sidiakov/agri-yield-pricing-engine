import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def generate_interactive_surface(
    model,
    sensitivity_beta: float,
    baseline_yield: float,
    commodity_name: str,
    current_yield_dev: Optional[float] = None,
) -> str:
    """
    Create and save a 3D price-risk surface.

    Parameters
    ----------
    model : any
        Placeholder for a yield model (not used directly here, but kept
        for compatibility with earlier versions).
    sensitivity_beta : float
        Percent price move per +1.0 in yield_deviation
        (e.g., -130 means +10% yield -> roughly -13% price move).
    baseline_yield : float
        Trend yield level (used for labeling only).
    commodity_name : str
        Human-readable commodity name.
    current_yield_dev : float, optional
        Scenario yield deviation to highlight with a marker.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    elasticity = abs(sensitivity_beta) / 100.0
    print(f"   > Calculating risk surface (elasticity: {elasticity:.2f}x)...")

    # X-axis: yield deviation from trend (-20% to +20%)
    x_vals = np.linspace(-0.20, 0.20, 81)

    # Y-axis: days within the growing/marketing window (10 to 180)
    y_vals = np.linspace(10, 180, 81)

    X, Y = np.meshgrid(x_vals, y_vals)

    # Time factor: 0 near the start, 1 near the end of the window
    time_factor = np.sqrt(Y / Y.max())

    # Price impact model: linear in yield deviation, scaled by horizon
    # Z_return is "expected price impact (%)"
    Z_return = sensitivity_beta * X * time_factor

    # Keep the surface within a realistic band so the plot is interpretable
    Z_return = np.clip(Z_return, -100.0, 150.0)

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z_return,
        colorscale="RdYlGn",
        colorbar=dict(title="Price impact (%)"),
        name="Price risk surface",
    )

    fig = go.Figure(data=[surface])

    # Optional: highlight the current scenario (where the engine thinks we are)
    if current_yield_dev is not None:
        current_x = float(current_yield_dev)
        current_y = float(np.median(y_vals))
        current_z = float(sensitivity_beta * current_x * np.sqrt(current_y / Y.max()))
        current_z = float(np.clip(current_z, -100.0, 150.0))

        fig.add_trace(
            go.Scatter3d(
                x=[current_x],
                y=[current_y],
                z=[current_z],
                mode="markers+text",
                marker=dict(size=6, color="yellow", symbol="diamond"),
                text=[f"Current: {current_x:+.2f} dev -> {current_z:+.1f}%"],
                textposition="top center",
                name="You are here",
            )
        )

    fig.update_layout(
        title=f"{commodity_name}: Yield vs Time -> Price Risk",
        scene=dict(
            xaxis_title="Yield deviation from trend",
            yaxis_title="Days in growing season",
            zaxis_title="Expected price impact (%)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save to disk
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    safe_name = (
        commodity_name.lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )
    output_path = os.path.join(out_dir, f"{safe_name}_risk_surface.html")
    fig.write_html(output_path, include_plotlyjs="cdn")

    return output_path


def print_executive_summary(
    commodity_name: str,
    r2_score: float,
    sensitivity_beta: float,
    risk_df: pd.DataFrame,
) -> None:
    """
    Print a concise, interview-ready narrative of the results.

    Parameters
    ----------
    commodity_name : str
        Name of the commodity (for display).
    r2_score : float
        In-sample R^2 of the yield model.
    sensitivity_beta : float
        Percent price move per +1.0 in yield_deviation.
    risk_df : DataFrame
        Historical table with columns:
        ['year', 'yield_deviation', 'harvest_return_pct'].
    """
    elasticity = abs(sensitivity_beta) / 100.0
    relationship = "inverse" if sensitivity_beta < 0 else "direct"

    print("\n" + "=" * 60)
    print(f"EXECUTIVE SUMMARY - {commodity_name}")
    print("=" * 60)

    # 1. Weather -> Yield signal strength
    if r2_score <= 0:
        print("1. WEATHER SIGNAL: Model could not extract a reliable signal from weather.")
    elif r2_score < 0.15:
        print(f"1. WEATHER SIGNAL: Weak but present (R^2 ~= {r2_score:.2f}).")
    elif r2_score < 0.35:
        print(f"1. WEATHER SIGNAL: Moderate explanatory power (R^2 ~= {r2_score:.2f}).")
    else:
        print(f"1. WEATHER SIGNAL: Strong structural link (R^2 ~= {r2_score:.2f}).")

    # 2. Yield -> Price elasticity
    print(
        f"\n2. YIELD -> PRICE ELASTICITY: {sensitivity_beta:+.1f}% per +1.0 dev "
        f"({elasticity:.2f}x {relationship} relationship)."
    )

    # 3. Historical sanity checks
    if risk_df is None or risk_df.empty:
        print("\n3. HISTORICAL REALITY: No sufficient futures history to cross-check.")
        print("\n" + "=" * 60 + "\n")
        return

    try:
        bad = risk_df[risk_df["yield_deviation"] < -0.05]
        good = risk_df[risk_df["yield_deviation"] > 0.05]

        avg_bad = bad["harvest_return_pct"].mean()
        avg_good = good["harvest_return_pct"].mean()

        print("\n3. HISTORICAL REALITY:")
        if not np.isnan(avg_bad):
            print(
                f"   - In low-yield years (<-5% dev), futures delivered "
                f"{avg_bad:+.1f}% on average."
            )
        if not np.isnan(avg_good):
            print(
                f"   - In high-yield years (>+5% dev), futures delivered "
                f"{avg_good:+.1f}% on average."
            )
    except Exception:
        print("\n3. HISTORICAL REALITY: Could not compute conditional returns cleanly.")

    print("\n" + "=" * 60 + "\n")
