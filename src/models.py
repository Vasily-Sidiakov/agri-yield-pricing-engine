import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_yield_model(features_df):
    """
    Trains a model to predict Yield DEVIATION based on Advanced Stress Features.
    """
    # Filter for years where we have valid deviation data
    data = features_df.dropna(subset=['yield_deviation'])
    
    if data.empty:
        # Fallback if no data (avoids crash)
        return LinearRegression(), 0.0

    # === UPDATE: USE ADVANCED FEATURES ===
    # 1. Heat (GDD)
    # 2. Water Availability (Soil Moisture is better than raw Rain)
    # 3. Flood Risk (Precip Squared - captures non-linear toxicity of too much rain)
    # 4. Plant Thirst (VPD - captures atmospheric drought)
    
    # Ensure these columns exist (handling potential first-run missing col issues)
    required_cols = ['weighted_gdd', 'weighted_soil', 'precip_sq', 'weighted_vpd']
    
    # Intersection check to be safe
    available_cols = [c for c in required_cols if c in data.columns]
    
    if not available_cols:
         # Fallback to simple features if advanced ones aren't processed yet
         available_cols = ['weighted_gdd', 'weighted_precip']

    X = data[available_cols]
    y = data['yield_deviation']
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    
    return model, r2

def analyze_price_risk(features_df, ticker):
    """
    Links Yield Deviations to Price Returns.
    Uses a 'Growing Season' window (May-Oct) to capture weather volatility.
    """
    # Load and clean price data
    try:
        price_df = pd.read_csv(f"data/raw/market_{ticker}.csv")
        # Ensure the date column is standard
        if 'Date' in price_df.columns:
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            price_df.set_index('Date', inplace=True)
        else:
            # If index is already date (yfinance update sometimes does this)
            price_df.index = pd.to_datetime(price_df.index)
    except Exception:
        return pd.DataFrame() # Return empty if file not found
    
    results = []
    
    for _, row in features_df.dropna(subset=['yield_deviation']).iterrows():
        year = int(row['year'])
        deviation = row['yield_deviation']
        
        # 1. Define Bucket based on Deviation
        if deviation < -0.05: # >5% below trend
            bucket = "Low Yield (Bullish)"
        elif deviation > 0.05: # >5% above trend
            bucket = "High Yield (Bearish)"
        else:
            bucket = "Normal Yield"
            
        try:
            # May 1 (Planting/Emergence) to Oct 1 (Harvest Start)
            start_date = f"{year}-05-01"
            end_date = f"{year}-10-01"
            
            # Get the price slice
            window = price_df.loc[start_date:end_date]
            
            if len(window) < 10: # Ensure enough data points
                continue
                
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
    """
    Calculates Beta: How much Price % changes for a 1% Yield Deviation.
    """
    if risk_df.empty: return 0.0
    
    df = risk_df.dropna()
    if len(df) < 5: return 0.0
    
    # X = Yield Deviation (e.g. -0.10)
    X = df['yield_deviation']
    # Y = Price Return (e.g., +20.0)
    y = df['harvest_return_pct']
    
    # Fit line
    slope, _ = np.polyfit(X, y, 1)
    return slope