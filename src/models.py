import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_config

def train_yield_model(features_df):
    """
    Trains an Adaptive Regime-Switching Model.
    Switches between Standard Regression and Auto-Regressive (AR-X) based on crop biology.
    """
    # Load config to check biology
    config = load_config()
    # Infer commodity from filename or pass it, but for now we detect biennial signal from data
    # We check if the 'is_biennial' flag would be useful, but here we can infer from the data structure
    
    data = features_df.dropna(subset=['yield_deviation', 'lag_1_yield_dev'])
    
    if data.empty:
        return LinearRegression(), 0.0

    # 1. DEFINE CANDIDATE FEATURES
    feature_pool = ['weighted_bio_growth', 'weighted_soil_moist', 'weighted_precip']
    
    # 2. CHECK FOR BIENNIAL SIGNAL (The "Paper" Logic)
    # Bernardes et al. (2012) state correlation between yield variation and previous yield.
    # We calculate autocorrelation of yield.
    autocorr = data['yield_deviation'].corr(data['lag_1_yield_dev'])
    
    is_biennial_detected = False
    if autocorr < -0.3: # Strong Negative Autocorrelation (High -> Low -> High)
        is_biennial_detected = True
        print(f"   > (DEBUG) Biennial Cycle Detected (Autocorr: {autocorr:.2f}). Activating AR-X Model.")
        feature_pool.append('lag_1_yield_dev') # Add Memory to the model
        
        # If biennial (Coffee), we also care about Frost (Accumulated Stress)
        feature_pool.append('weighted_acc_stress') 
        
    else:
        # Standard Annual Crop Logic (Corn/Soy)
        if data['stress_x_dryness'].sum() > 1.0:
            feature_pool.append('stress_x_dryness')
            feature_pool.append('precip_sq')

    # 3. ROBUST MODELING (RIDGE)
    available_cols = [c for c in feature_pool if c in data.columns]
    
    X = data[available_cols]
    y = data['yield_deviation']
    
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    r2 = model.score(X, y)
    
    return model, r2

def analyze_price_risk(features_df, ticker):
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
