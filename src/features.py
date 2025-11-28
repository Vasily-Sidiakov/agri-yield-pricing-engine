import pandas as pd
import numpy as np
from src.utils import load_config

def calculate_gdd(tmax, tmin, base_temp=10):
    """Calculates Growing Degree Days (Celsius)."""
    avg_temp = (tmax + tmin) / 2
    return np.maximum(0, avg_temp - base_temp)

def load_yield_history(commodity_key):
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
            
            # Calculations
            df['gdd'] = calculate_gdd(df['tmax'], df['tmin'])
            df['extreme_heat'] = (df['tmax'] > 30).astype(int)
            
            # === THE FIX: CROSS-YEAR SEASON LOGIC ===
            if start_month <= end_month:
                # Standard Summer Crop (e.g. Corn: April -> Oct)
                mask = (df['date'].dt.month >= start_month) & (df['date'].dt.month <= end_month)
                season_df = df[mask].copy()
                season_df['crop_year'] = season_df['date'].dt.year
            else:
                # Winter/Tropical Crop (e.g. Coffee: Oct -> May)
                # Logic: Month is >= Start OR Month is <= End
                mask = (df['date'].dt.month >= start_month) | (df['date'].dt.month <= end_month)
                season_df = df[mask].copy()
                
                # Critical: Shift the "Crop Year" for the early months
                # If it's Oct/Nov/Dec 2020, it belongs to the 2021 Harvest
                season_df['crop_year'] = season_df['date'].dt.year
                season_df.loc[season_df['date'].dt.month >= start_month, 'crop_year'] += 1
            # ========================================
            
            # Aggregate by the new 'crop_year'
            yearly_stats = season_df.groupby('crop_year').agg({
                'gdd': 'sum',
                'precip': 'sum',
                'vpd': 'mean',
                'soil_moist': 'mean',
                'extreme_heat': lambda x: x[season_df.loc[x.index, 'date'].dt.month == critical_month].sum()
            }).reset_index()
            
            yearly_stats.columns = ['year', 'gdd', 'precip', 'vpd', 'soil_moist', 'heat_stress_days']
            
            # Vectorized Weighting
            w = region['weight']
            yearly_stats['weighted_gdd'] = yearly_stats['gdd'] * w
            yearly_stats['weighted_precip'] = yearly_stats['precip'] * w
            yearly_stats['weighted_vpd'] = yearly_stats['vpd'] * w
            yearly_stats['weighted_soil'] = yearly_stats['soil_moist'] * w
            yearly_stats['weighted_heat_stress'] = yearly_stats['heat_stress_days'] * w
            
            all_years_data.append(yearly_stats)
        except FileNotFoundError:
            continue

    if not all_years_data:
        return None

    combined_df = pd.concat(all_years_data)
    
    # Global Average
    final_features = combined_df.groupby('year')[['weighted_gdd', 'weighted_precip', 'weighted_vpd', 'weighted_soil', 'weighted_heat_stress']].sum().reset_index()

    # Quadratic Rain
    final_features['precip_sq'] = final_features['weighted_precip'] ** 2

    # Attach Yield
    final_features['usda_yield'] = final_features['year'].map(historical_yields)
    
    # Detrending
    valid_data = final_features.dropna(subset=['usda_yield'])
    if len(valid_data) > 5:
        z = np.polyfit(valid_data['year'], valid_data['usda_yield'], 1)
        p = np.poly1d(z)
        final_features['trend_yield'] = p(final_features['year'])
        final_features['yield_deviation'] = (final_features['usda_yield'] - final_features['trend_yield']) / final_features['trend_yield']
    else:
        final_features['yield_deviation'] = 0.0

    output_path = f"data/processed/features_{commodity_key}.csv"
    final_features.to_csv(output_path, index=False)
    
    return final_features
