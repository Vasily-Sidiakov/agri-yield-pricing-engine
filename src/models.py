import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_yield_model(features_df):
    """
    Trains an Adaptive Regime-Switching Model.
    Automatically selects the best features based on the data signature.
    """
    data = features_df.dropna(subset=['yield_deviation'])
    
    if data.empty:
        return LinearRegression(), 0.0

    # 1. DEFINE CANDIDATE FEATURES
    # We have a pool of advanced metrics. We need to pick the ones that matter.
    # Base: Growth + Water
    feature_pool = ['weighted_bio_growth', 'weighted_soil_moist', 'weighted_precip']
    
    # Check if this crop experienced specific stress events
    # If the sum of 'stress_x_dryness' is high, it's likely a Heat/Drought crop (Corn)
    if data['stress_x_dryness'].sum() > 1.0:
        feature_pool.append('stress_x_dryness') # Interaction Term
        feature_pool.append('precip_sq')        # Flood Risk
        
    # If 'weighted_acc_stress' is high but 'stress_x_dryness' is low, 
    # it might be pure temp stress (Frost for Coffee)
    elif data['weighted_acc_stress'].sum() > 0.1:
        feature_pool.append('weighted_acc_stress') # Pure Thermal Shock (Frost)

    # 2. ROBUST MODELING (RIDGE REGRESSION)
    # Because we have complex, correlated features (like Rain and Soil Moisture),
    # a standard Linear Regression might overfit. 
    # We use Ridge Regression (L2 Regularization) to handle multicollinearity.
    
    # Filter for existing columns only
    available_cols = [c for c in feature_pool if c in data.columns]
    
    X = data[available_cols]
    y = data['yield_deviation']
    
    # Pipeline: Scale Data -> Ridge Regression
    # Scaling is crucial when mixing units (Degrees vs Millimeters)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    r2 = model.score(X, y)
    
    # Extract the internal linear model for coefficient analysis if needed
    final_model = model.named_steps['ridge']
    
    return final_model, r2

def analyze_price_risk(features_df, ticker):
    """
    Standard Price Risk Analysis (Growing Season Window).
    """
    try:
        price_df = pd.read_csv(f"data/raw/market_{ticker}.csv")
        if 'Date' in price_df.columns:
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            price_df.set_index('Date', inplace=True)
        else:
            price_df.index = pd.to_datetime(price_df.index)
    except Exception:
        return pd.DataFrame()
    
    results = []
    
    for _, row in features_df.dropna(subset=['yield_deviation']).iterrows():
        year = int(row['year'])
        deviation = row['yield_deviation']
        
        # Bucket Logic
        if deviation < -0.05: bucket = "Low Yield (Bullish)"
        elif deviation > 0.05: bucket = "High Yield (Bearish)"
        else: bucket = "Normal Yield"
            
        try:
            # Dynamic Season Window
            # If crop_year != year (Winter crop), we adjust the window
            # Simple heuristic: Look at the 6 months leading up to harvest
            # Since we don't pass harvest date here, we default to May-Oct for now
            # In a V5.0, we would pass harvest_month dynamically.
            start_date = f"{year}-05-01"
            end_date = f"{year}-10-01"
            
            window = price_df.loc[start_date:end_date]
            if len(window) < 10: continue
                
            p_start = window.iloc[0]['price']
            p_end = window.iloc[-1]['price']
            pct_return = ((p_end - p_start) / p_start) * 100
            
            results.append({
                'year': year,
                'yield_bucket': bucket,
                'yield_deviation': deviation,
                'harvest_return_pct': pct_return
            })
        except Exception:
            continue
            
    return pd.DataFrame(results)

def calculate_price_sensitivity(risk_df):
    if risk_df.empty: return 0.0
    df = risk_df.dropna()
    if len(df) < 5: return 0.0
    
    slope, _ = np.polyfit(df['yield_deviation'], df['harvest_return_pct'], 1)
    return slope
