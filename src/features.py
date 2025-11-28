import pandas as pd
import numpy as np
from src.utils import load_config

# ==============================================================================
#  AGRONOMIC CONSTANTS
# ==============================================================================
CROP_PROFILES = {
    'corn_us':      {'t_base': 10, 't_opt': 30, 't_max': 45, 'frost_sensitive': False, 'heat_sensitive': True},
    'soybeans_us':  {'t_base': 10, 't_opt': 29, 't_max': 40, 'frost_sensitive': False, 'heat_sensitive': True},
    'wheat_us':     {'t_base': 0,  't_opt': 22, 't_max': 35, 'frost_sensitive': False, 'heat_sensitive': True}, 
    'coffee_br':    {'t_base': 15, 't_opt': 24, 't_max': 32, 'frost_sensitive': True,  'heat_sensitive': False}, 
    'cocoa_iv':     {'t_base': 18, 't_opt': 25, 't_max': 33, 'frost_sensitive': False, 'heat_sensitive': True},
    'sugar_br':     {'t_base': 12, 't_opt': 32, 't_max': 45, 'frost_sensitive': False, 'heat_sensitive': False}, 
    'cotton_us':    {'t_base': 15, 't_opt': 32, 't_max': 40, 'frost_sensitive': False, 'heat_sensitive': True},
    'rice_us':      {'t_base': 12, 't_opt': 30, 't_max': 42, 'frost_sensitive': False, 'heat_sensitive': True},
}

# ==============================================================================
#  MATH FUNCTIONS
# ==============================================================================
def calculate_beta_function_thermal_time(t_mean, t_base, t_opt, t_max):
    if t_opt <= t_base or t_max <= t_opt: return 0.0
    t_mean = np.clip(t_mean, t_base, t_max)
    exponent = (t_max - t_opt) / (t_opt - t_base)
    term1 = ((t_max - t_mean) / (t_max - t_opt)) * ((t_mean - t_base) / (t_opt - t_base)) ** exponent
    return np.maximum(0, term1)

def load_yield_history(commodity_key):
    try:
        df = pd.read_csv("data/yield_history.csv")
        subset = df[df['commodity'] == commodity_key]
        return pd.Series(subset.yield_value.values, index=subset.year).to_dict()
    except Exception as e:
        print(f"Error loading yield history: {e}")
        return {}

# ==============================================================================
#  MAIN PROCESSING ENGINE
# ==============================================================================
def process_weather_features(commodity_key):
    print(f"   > (DEBUG) Running Bio-Adaptive Agronomy Engine for {commodity_key}...")
    config = load_config()
    commodity = config['commodities'][commodity_key]
    profile = CROP_PROFILES.get(commodity_key, CROP_PROFILES['corn_us']) 
    
    historical_yields = load_yield_history(commodity_key)
    if not historical_yields: return None

    start_month = commodity['seasons']['start_month']
    end_month = commodity['seasons']['end_month']
    critical_month = commodity['seasons'].get('critical_heat_month', 7)
    
    all_years_data = []

    for region in commodity['regions']:
        file_path = f"data/raw/weather_{region['name']}.csv"
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df['t_mean'] = (df['tmax'] + df['tmin']) / 2
            
            # 1. Non-Linear Growth
            df['bio_growth_days'] = calculate_beta_function_thermal_time(
                df['t_mean'].values, profile['t_base'], profile['t_opt'], profile['t_max']
            )
            
            # 2. Stress Logic
            if profile['heat_sensitive']:
                df['stress_event'] = np.maximum(0, df['tmax'] - 32) 
            elif profile['frost_sensitive']:
                # Frost Logic (Bernards et al., 2012 suggests extreme min temp effects)
                df['stress_event'] = np.maximum(0, 2 - df['tmin']) 
            else:
                df['stress_event'] = 0.0

            # 3. Season Filtering
            if profile['frost_sensitive']:
                # Perennial / Biennial Logic
                season_df = df.copy()
                season_df['crop_year'] = season_df['date'].dt.year
                season_df.loc[season_df['date'].dt.month >= 4, 'crop_year'] += 1
            else:
                # Annual Logic
                if start_month <= end_month:
                    mask = (df['date'].dt.month >= start_month) & (df['date'].dt.month <= end_month)
                    season_df = df[mask].copy()
                    season_df['crop_year'] = season_df['date'].dt.year
                else:
                    mask = (df['date'].dt.month >= start_month) | (df['date'].dt.month <= end_month)
                    season_df = df[mask].copy()
                    season_df['crop_year'] = season_df['date'].dt.year
                    season_df.loc[season_df['date'].dt.month >= start_month, 'crop_year'] += 1
            
            # 4. Aggregation
            yearly_stats = season_df.groupby('crop_year').agg({
                'bio_growth_days': 'sum',
                'precip': 'sum',
                'vpd': 'mean',
                'soil_moist': 'mean',
                'stress_event': 'sum'
            }).reset_index()
            
            yearly_stats.columns = ['year', 'bio_growth', 'precip', 'vpd', 'soil_moist', 'acc_stress']
            
            # Weighting
            w = region['weight']
            for col in ['bio_growth', 'precip', 'vpd', 'soil_moist', 'acc_stress']:
                yearly_stats[f'weighted_{col}'] = yearly_stats[col] * w
            
            all_years_data.append(yearly_stats)
            
        except FileNotFoundError:
            continue

    if not all_years_data: return None

    # Combine
    combined_df = pd.concat(all_years_data)
    final_features = combined_df.groupby('year')[[
        'weighted_bio_growth', 'weighted_precip', 'weighted_vpd', 
        'weighted_soil_moist', 'weighted_acc_stress'
    ]].sum().reset_index()

    # Features
    final_features['stress_x_dryness'] = final_features['weighted_acc_stress'] * final_features['weighted_vpd']
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
        
        # === NEW: LAGGED YIELD (BIENNIAL MEMORY) ===
        # We shift the yield deviation by 1 year.
        # This tells the model: "What was the yield last year?"
        final_features['lag_1_yield_dev'] = final_features['yield_deviation'].shift(1)
    else:
        final_features['yield_deviation'] = 0.0
        final_features['lag_1_yield_dev'] = 0.0

    output_path = f"data/processed/features_{commodity_key}.csv"
    final_features.to_csv(output_path, index=False)
    
    return final_features
