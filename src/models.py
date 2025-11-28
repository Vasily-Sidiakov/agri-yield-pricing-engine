import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_yield_model(features_df):
    """
    Trains a Robust 'Kitchen Sink' Model using Ridge Regression.
    Instead of guessing which features to use, we feed all advanced metrics
    (Frost, Memory, Heat, Drought) and let the L2 Regularization sort it out.
    """
    # Filter for valid data
    # We need rows where we have the 'Yield Deviation' target
    data = features_df.dropna(subset=['yield_deviation'])
    
    if data.empty:
        return Ridge(), 0.0

    # 1. ASSEMBLE THE FEATURE ARSENAL
    # We want to give the model every possible tool to find the signal.
    
    potential_features = [
        # Base Growth
        'weighted_bio_growth', 
        'weighted_precip',
        
        # Water Stress (Corn/Soy/Wheat)
        'weighted_soil_moist',
        'stress_x_dryness',     # The "Multiplier" (Heat * Drought)
        'precip_sq',            # Flood risk
        
        # Thermal Stress (Coffee/Fruit)
        'weighted_acc_stress',  # Accumulator (Frost or Heat Degree Days)
        
        # Biennial Memory (Coffee/Trees)
        # Based on Bernardes et al. (2012): Previous year yield impacts current year
        'lag_1_yield_dev'       
    ]
    
    # 2. SELECT AVAILABLE FEATURES
    # We only use columns that actually exist in the dataframe 
    # (e.g. 'lag_1_yield_dev' is only created if data depth allows)
    X_cols = [c for c in potential_features if c in data.columns]
    
    # Drop rows that have NaNs in these specific feature columns
    # (e.g. The first year of data won't have a Lag-1 value)
    model_data = data.dropna(subset=X_cols)
    
    if len(model_data) < 5:
        print("   > (Warning) Not enough data depth for advanced modeling.")
        return Ridge(), 0.0

    X = model_data[X_cols]
    y = model_data['yield_deviation']
    
    # 3. TRAIN RIDGE REGRESSION
    # We use Ridge (L2) because these features are highly correlated.
    # Ridge prevents "over-reacting" to any single variable.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    r2 = model.score(X, y)
    
    return model, r2

def analyze_price_risk(features_df, ticker):
    """
    Links Yield Deviations to Price Returns.
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
        
        if deviation < -0.05: bucket = "Low Yield (Bullish)"
        elif deviation > 0.05: bucket = "High Yield (Bearish)"
        else: bucket = "Normal Yield"
            
        try:
            # Dynamic Window: May to Oct (Harvest Season)
            # In a production environment, this dates range would be dynamic per crop
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
