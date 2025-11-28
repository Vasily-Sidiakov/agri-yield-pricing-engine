import pandas as pd
import numpy as np
from src.utils import load_config

def calculate_gdd(tmax, tmin, base_temp=10):
    """Calculates Growing Degree Days (Celsius)."""
    avg_temp = (tmax + tmin) / 2
    return np.maximum(0, avg_temp - base_temp)

def load_yield_history(commodity_key):
    """Loads historical yields from the CSV file."""
    try:
        df = pd.read_csv("data/yield_history.csv")
        subset = df[df['commodity'] == commodity_key]
        return pd.Series(subset.yield_value.values, index=subset.year).to_dict()
    except Exception as e:
        print(f"Error loading yield history: {e}")
        return {}

def process_weather_features(commodity_key):
    config = load_config()
    commodity = config['commodities'][commodity_key]
    
    historical_yields = load_yield_history(commodity_key)
    if not historical_yields:
        print(f"Error: No yield history found for {commodity_key}")
        return None

    start_month = commodity['seasons']['start_month']
    end_month = commodity['seasons']['end_month']
    critical_month = commodity['seasons'].get('critical_heat_month', 7)
    
    all_years_data = []

    for region in commodity['regions']:
        file_path = f"data/raw/weather_{region['name']}.csv"
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Basic Calculations
            df['gdd'] = calculate_gdd(df['tmax'], df['tmin'])
            df['extreme_heat'] = (df['tmax'] > 30).astype(int)
            
            # Filter for Growing Season
            mask_season = (df['date'].dt.month >= start_month) & (df['date'].dt.month <= end_month)
            season_df = df[mask_season].copy()
            season_df['year'] = season_df['date'].dt.year
            
            # Aggregation: Sum for accumulations, Mean for state variables (Soil/VPD)
            yearly_stats = season_df.groupby('year').agg({
                'gdd': 'sum',
                'precip': 'sum',
                'vpd': 'mean',          # NEW: Average atmospheric thirst
                'soil_moist': 'mean',   # NEW: Average root zone moisture
                'extreme_heat': lambda x: x[season_df.loc[x.index, 'date'].dt.month == critical_month].sum()
            }).reset_index()
            
            yearly_stats.columns = ['year', 'gdd', 'precip', 'vpd', 'soil_moist', 'heat_stress_days']
            yearly_stats['weight'] = region['weight']
            
            all_years_data.append(yearly_stats)
        except FileNotFoundError:
            continue

    if not all_years_data:
        return None

    combined_df = pd.concat(all_years_data)
    
    # Weighted Average of Regions
    final_features = combined_df.groupby('year').apply(
        lambda x: pd.Series({
            'weighted_gdd': np.average(x['gdd'], weights=x['weight']),
            'weighted_precip': np.average(x['precip'], weights=x['weight']),
            'weighted_vpd': np.average(x['vpd'], weights=x['weight']),             # NEW
            'weighted_soil': np.average(x['soil_moist'], weights=x['weight']),     # NEW
            'weighted_heat_stress': np.average(x['heat_stress_days'], weights=x['weight'])
        }),
        include_groups=False 
    ).reset_index()

    # === NEW: Quadratic Term for Rain (The "Goldilocks" curve) ===
    # Captures the fact that too much rain (flooding) is bad
    final_features['precip_sq'] = final_features['weighted_precip'] ** 2

    # Attach Raw Yield
    final_features['usda_yield'] = final_features['year'].map(historical_yields)
    
    # Detrending Logic
    valid_data = final_features.dropna(subset=['usda_yield'])
    if len(valid_data) > 5:
        z = np.polyfit(valid_data['year'], valid_data['usda_yield'], 1)
        p = np.poly1d(z)
        final_features['trend_yield'] = p(final_features['year'])
        
        # Calculate Deviation
        final_features['yield_deviation'] = (final_features['usda_yield'] - final_features['trend_yield']) / final_features['trend_yield']
    else:
        final_features['yield_deviation'] = 0.0

    output_path = f"data/processed/features_{commodity_key}.csv"
    final_features.to_csv(output_path, index=False)
    
    return final_features