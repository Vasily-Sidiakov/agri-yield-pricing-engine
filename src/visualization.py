import numpy as np
import pandas as pd
import plotly.graph_objects as go

def generate_interactive_surface(model, sensitivity_beta, baseline_yield, commodity_name, current_yield_dev=None):
    """
    Generates a 3D Risk Surface with a YELLOW 'You Are Here' Marker.
    Cleanest Version: No static legend, just interactive tooltips.
    """
    # --- 1. THE MATH ---
    elasticity = abs(sensitivity_beta) / 100
    print(f"   > Calculating 'Neon Prism' Surface (Elasticity: {elasticity:.2f}x)...")

    # Grid Generation
    x_range = np.linspace(-0.20, 0.20, 100) 
    y_range = np.linspace(10, 180, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Z-Axis Model
    time_factor = np.sqrt(Y / 180.0) 
    Z = (X * sensitivity_beta * time_factor)

    # --- 2. THE VISUALS: SURFACE & MARKER ---
    
    # Custom "Vaporwave" Gradient
    vibrant_colors = [
        [0.0, '#00F0FF'], # Cyan
        [0.5, '#7D00FF'], # Purple
        [1.0, '#FF00AA']  # Pink
    ]
    
    # 2a. The Main Surface
    surface_trace = go.Surface(
        z=Z, x=x_range, y=Y,
        colorscale=vibrant_colors,
        opacity=1.0, 
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        contours_x=dict(show=True, start=-0.20, end=0.20, size=0.02, color='rgba(255,255,255,0.4)', width=1),
        contours_y=dict(show=True, start=10, end=180, size=10, color='rgba(255,255,255,0.4)', width=1),
        contours_z=dict(show=True, usecolormap=False, color='#333', width=2, project_z=True),
        hovertemplate = "<b>Yield: %{x:.0%}</b><br>Horizon: %{y:.0f} Days<br><b>Price: %{z:.1f}%</b><extra></extra>",
        showscale=False # No color bar
    )
    
    data_traces = [surface_trace]

    # 2b. The "Current State" Marker (The Overlay)
    if current_yield_dev is not None:
        sim_time = 90 
        sim_time_factor = np.sqrt(sim_time / 180.0)
        sim_price_impact = current_yield_dev * sensitivity_beta * sim_time_factor
        
        # Create the YELLOW Glowing Orb
        marker_trace = go.Scatter3d(
            x=[current_yield_dev],
            y=[sim_time],
            z=[sim_price_impact],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#FFFF00',  # ELECTRIC YELLOW (High Contrast)
                line=dict(color='white', width=1), 
                opacity=1.0
            ),
            # Only show the percentage number floating above the ball
            text=[f"{sim_price_impact:+.1f}%"], 
            textposition="top center",
            textfont=dict(family="Arial", size=14, color="#FFFF00", weight="bold"),
            
            # Hide from static legend
            showlegend=False,
            
            # Detailed Hover Tooltip
            hovertemplate = (
                "<b>SCENARIO SIMULATION</b><br>" +
                "Yield Input: %{x:.1%}<br>" +
                "Horizon: %{y:.0f} Days<br>" +
                "<b>Predicted Price: %{z:.1f}%</b>" +
                "<extra></extra>"
            )
        )
        
        # The Drop Line
        drop_line = go.Scatter3d(
            x=[current_yield_dev, current_yield_dev],
            y=[sim_time, sim_time],
            z=[sim_price_impact, -30], 
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        )
        
        data_traces.append(drop_line)
        data_traces.append(marker_trace)

    fig = go.Figure(data=data_traces)

    # --- 3. THE LAYOUT ---
    
    # Fonts & Colors
    title_font = "Didot, Georgia, serif"
    axis_font = "-apple-system, BlinkMacSystemFont, Roboto, sans-serif"
    bg_color = "#000000"
    text_color = "#FFFFFF" 
    grid_color = "#333333" 

    # Subtitle Logic
    if sensitivity_beta < 0:
        subtitle = f"SENSITIVITY: 1% YIELD DROP ≈ {elasticity:.2f}% PRICE RALLY"
    else:
        subtitle = f"SENSITIVITY: 1% YIELD RISE ≈ {elasticity:.2f}% PRICE RALLY"

    fig.update_layout(
        title={
            'text': f"<b>{commodity_name.upper()}</b><br>"
                    f"<span style='font-size: 14px; font-family: {axis_font}; color: #aaa; letter-spacing: 1px;'>{subtitle}</span>",
            'y':0.90, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(family=title_font, size=36, color=text_color)
        },
        scene={
            'xaxis': {
                'title': 'YIELD DEVIATION', 'tickformat': '.0%',
                'backgroundcolor': bg_color, 'gridcolor': grid_color, 'showbackground': True,
                'title_font': {"family": axis_font, "size": 11, "color": "#888"},
                'tickfont': {"family": axis_font, "size": 10, "color": "#666"}
            },
            'yaxis': {
                'title': 'HORIZON (DAYS)',
                'backgroundcolor': bg_color, 'gridcolor': grid_color, 'showbackground': True,
                'title_font': {"family": axis_font, "size": 11, "color": "#888"},
                'tickfont': {"family": axis_font, "size": 10, "color": "#666"}
            },
            'zaxis': {
                'title': 'PRICE (%)',
                'backgroundcolor': bg_color, 'gridcolor': grid_color, 'showbackground': True,
                'title_font': {"family": axis_font, "size": 11, "color": "#888"},
                'tickfont': {"family": axis_font, "size": 10, "color": "#666"}
            },
            'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 0.5}},
            'dragmode': 'orbit'
        },
        paper_bgcolor=bg_color,
        font=dict(family=axis_font, color=text_color),
        margin=dict(l=0, r=0, b=0, t=100), # Bottom margin reduced since legend is gone
        showlegend=False
    )

    safe_name = commodity_name.replace(" ", "_")
    output_path = f"surface_{safe_name}.html"
    fig.write_html(output_path)
    print(f"   > Interactive Chart saved to {output_path}")
    return output_path

def print_executive_summary(commodity_name, r2_score, sensitivity, risk_df):
    """
    Simplified Executive Summary.
    """
    print(f"\n{'='*60}")
    print(f"   RISK PROFILE: {commodity_name.upper()}")
    print(f"{'='*60}")

    elasticity = abs(sensitivity) / 100
    direction = "Opposite" if sensitivity < 0 else "Parallel"
    
    print(f"\n1. THE 'CHEAT SHEET'")
    print(f"   > Elasticity Ratio: {elasticity:.1f}x")
    print(f"   > Interpretation: If Yield moves 1%, Price moves {elasticity:.1f}% in the {direction} direction.")

    confidence = "Low"
    if r2_score > 0.4: confidence = "Moderate"
    if r2_score > 0.6: confidence = "High"
    print(f"\n2. WEATHER SIGNAL STRENGTH: {confidence} (R² = {r2_score:.2f})")
    
    try:
        bad_years = risk_df[risk_df['yield_deviation'] < -0.05]
        avg_bad_return = bad_years['harvest_return_pct'].mean()
        
        print(f"\n3. HISTORICAL REALITY")
        if not np.isnan(avg_bad_return):
            print(f"   > When Yields actually crashed >5%, prices moved: {avg_bad_return:+.1f}%")
        else:
            print("   > No historical crashes >5% found in dataset.")
    except Exception:
        pass

    print(f"\n{'='*60}\n")
