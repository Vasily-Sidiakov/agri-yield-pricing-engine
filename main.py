from src.utils import load_config
from src.data_loader import fetch_market_data, fetch_weather_data
from src.stochastic import StochasticWeatherGenerator
from src.biophysics import BiophysicalTwin
from src.pricing import StochasticPricingEngine
from src.visualization import generate_interactive_surface, print_executive_summary
import sys
import webbrowser
import os
import numpy as np
import pandas as pd
import warnings

# Suppress the specific RankWarning from numpy if variance is low
warnings.simplefilter('ignore', np.RankWarning)

def main():
    print("==========================================")
    print("   AGRI-YIELD PRICING ENGINE (v4.1)       ")
    print("   (Institutional Stochastic Edition)     ")
    print("==========================================")
    
    config = load_config()
    if not config: sys.exit("Failed to load configuration.")
    
    # --- 1. MENU ---
    alias_map = {}
    print("\nAvailable Commodities:")
    for key, data in config['commodities'].items():
        print(f" - {data['name']}")
        simple_name = key.split('_')[0] 
        alias_map[simple_name] = key
        alias_map[key] = key
        clean_name = data['name'].lower().split(' (')[0]
        alias_map[clean_name] = key

    print("\n" + "-"*40)
    user_input = input("Which commodity would you like to analyze? ").strip().lower()
    
    if user_input in alias_map:
        selected_key = alias_map[user_input]
    else:
        print(f"\nSorry, I didn't recognize '{user_input}'.")
        sys.exit()
        
    comm_config = config['commodities'][selected_key]
    print(f"\n--- Initiating Analysis for {comm_config['name']} ---")

    # --- 2. DATA INGESTION ---
    print(f"   > Ingesting 44 Years of Satellite Data...")
    
    market_df = fetch_market_data(comm_config['ticker'], start_date="1980-01-01")
    current_price = market_df['price'].iloc[-1]
    
    weather_df = None
    for region in comm_config['regions']:
        w_df = fetch_weather_data(
            region['latitude'], region['longitude'], 
            "1980-01-01", "2024-12-31", region['name']
        )
        if weather_df is None: weather_df = w_df
        else: weather_df = pd.concat([weather_df, w_df]) 

    # --- 3. STOCHASTIC WEATHER GENERATION (Phase 1) ---
    print("\n--- Phase 1: Stochastic Weather Simulation ---")
    weather_gen = StochasticWeatherGenerator(selected_key)
    
    weather_gen.calibrate(weather_df, variable='tmax')
    weather_gen.calibrate(weather_df, variable='vpd')
    
    print("   > Running Monte Carlo Simulation (10,000 Paths)...")
    tmax_paths = weather_gen.simulate('2024-01-01', days_ahead=180, variable='tmax') 
    vpd_paths = weather_gen.simulate('2024-01-01', days_ahead=180, variable='vpd')
    
    aad_paths = weather_gen.calculate_accumulated_stress(vpd_paths)

    # --- 4. BIOPHYSICAL DIGITAL TWIN (Phase 2) ---
    print("\n--- Phase 2: Biophysical Crop Modeling ---")
    print("   > Solving Differential Equations (Biomass/Soil Moisture)...")
    
    bio_engine = BiophysicalTwin(selected_key)
    yield_paths, ks_history = bio_engine.solve_odes(tmax_paths, days_ahead=180)
    
    # Calculate Deviations
    baseline_simulated_yield = np.mean(yield_paths)
    
    # Safety check for zero yield (failed crop)
    if baseline_simulated_yield < 1e-6:
        baseline_simulated_yield = 1.0 
        
    yield_deviations = (yield_paths - baseline_simulated_yield) / baseline_simulated_yield

    # --- 5. FINANCIAL PRICING ENGINE (Phase 3) ---
    print("\n--- Phase 3: Gibson-Schwartz Pricing ---")
    print(f"   > Pricing Options based on Spot: ${current_price:.2f}")
    
    pricing_engine = StochasticPricingEngine(current_price)
    
    biomass_proxy = np.outer(np.linspace(0, 1, 180), yield_paths)
    
    price_paths, convenience_yields = pricing_engine.simulate_pricing_paths(
        biomass_proxy, aad_paths, days_ahead=180
    )
    
    final_prices = price_paths[-1, :]
    returns_pct = ((final_prices - current_price) / current_price) * 100
    
    # === SAFETY FIX: CALCULATE ELASTICITY ===
    # Check if there is enough variance in yield to calculate a slope
    yield_variance = np.std(yield_deviations)
    
    if yield_variance < 1e-6:
        # If yields are identical across all paths (Crop is robust/Weather invariant)
        sensitivity = 0.0
        print("   > Note: Yield variance is near zero (Stable Crop). Elasticity defaulting to 0.")
    else:
        # Perform regression
        sensitivity = np.polyfit(yield_deviations, returns_pct, 1)[0]
    
    elasticity = abs(sensitivity) / 100
    relationship = "INVERSE" if sensitivity < 0 else "DIRECT"
    
    print(f"\n   > Implied Market Elasticity: {elasticity:.2f}x ({relationship})")
    
    # --- 6. VISUALIZATION ---
    print("\n--- Generating 3D Risk Surface ---")
    
    print("\n" + "-"*40)
    print("   SCENARIO SIMULATOR")
    sim_input = input("   Enter Scenario (e.g. -0.10): ").strip()
    current_yield_dev = float(sim_input) if sim_input else None

    baseline = comm_config.get('baseline_yield', 100)
    
    file_path = generate_interactive_surface(
        None, 
        sensitivity, 
        baseline, 
        comm_config['name'],
        current_yield_dev=current_yield_dev
    )
    
    full_path = "file://" + os.path.abspath(file_path)
    print(f"   > Opening {file_path} in browser...")
    webbrowser.open(full_path)
    
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
