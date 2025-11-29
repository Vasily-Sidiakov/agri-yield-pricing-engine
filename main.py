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

def main():
    print("==========================================")
    print("   AGRI-YIELD PRICING ENGINE (v4.0)       ")
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
    
    # Get Market Data
    market_df = fetch_market_data(comm_config['ticker'], start_date="1980-01-01")
    current_price = market_df['price'].iloc[-1]
    
    # Get Weather Data (Force download to ensure we have VPD/Soil columns)
    weather_df = None
    for region in comm_config['regions']:
        w_df = fetch_weather_data(
            region['latitude'], region['longitude'], 
            "1980-01-01", "2024-12-31", region['name']
        )
        if weather_df is None: weather_df = w_df
        else: weather_df = pd.concat([weather_df, w_df]) # Simple stack for calibration

    # --- 3. STOCHASTIC WEATHER GENERATION (Phase 1) ---
    print("\n--- Phase 1: Stochastic Weather Simulation ---")
    weather_gen = StochasticWeatherGenerator(selected_key)
    
    # Calibrate OU Process
    weather_gen.calibrate(weather_df, variable='tmax')
    weather_gen.calibrate(weather_df, variable='vpd')
    
    # Simulate 10,000 Paths
    print("   > Running Monte Carlo Simulation (10,000 Paths)...")
    # Simulate next 180 days (Growing Season)
    tmax_paths = weather_gen.simulate('2024-01-01', days_ahead=180, variable='tmax') 
    vpd_paths = weather_gen.simulate('2024-01-01', days_ahead=180, variable='vpd')
    
    # Calculate Accumulated Stress (Integral of VPD)
    aad_paths = weather_gen.calculate_accumulated_stress(vpd_paths)

    # --- 4. BIOPHYSICAL DIGITAL TWIN (Phase 2) ---
    print("\n--- Phase 2: Biophysical Crop Modeling ---")
    print("   > Solving Differential Equations (Biomass/Soil Moisture)...")
    
    bio_engine = BiophysicalTwin(selected_key)
    # Solve ODEs
    yield_paths, ks_history = bio_engine.solve_odes(tmax_paths, days_ahead=180)
    
    # Calculate Yield Deviation % for each path
    # Assume baseline yield is the mean of our simulation (for relative pricing)
    baseline_simulated_yield = np.mean(yield_paths)
    yield_deviations = (yield_paths - baseline_simulated_yield) / baseline_simulated_yield

    # --- 5. FINANCIAL PRICING ENGINE (Phase 3) ---
    print("\n--- Phase 3: Gibson-Schwartz Pricing ---")
    print(f"   > Pricing Options based on Spot: ${current_price:.2f}")
    
    pricing_engine = StochasticPricingEngine(current_price)
    
    # Run Heston + Gibson-Schwartz Model
    # We pass the biomass (yield) paths to drive convenience yield
    # We pass AAD paths to drive volatility
    # We reshape yield_paths to match time steps (simple constant growth assumption for mapping)
    biomass_proxy = np.outer(np.linspace(0, 1, 180), yield_paths)
    
    price_paths, convenience_yields = pricing_engine.simulate_pricing_paths(
        biomass_proxy, aad_paths, days_ahead=180
    )
    
    # Calculate Final Returns for each path
    final_prices = price_paths[-1, :]
    returns_pct = ((final_prices - current_price) / current_price) * 100
    
    # Calculate Elasticity (Beta) from the simulated universe
    # Regress Simulated Returns vs Simulated Yield Deviations
    sensitivity = np.polyfit(yield_deviations, returns_pct, 1)[0]
    
    elasticity = abs(sensitivity) / 100
    relationship = "INVERSE" if sensitivity < 0 else "DIRECT"
    
    print(f"\n   > Implied Market Elasticity: {elasticity:.2f}x ({relationship})")
    
    # --- 6. VISUALIZATION ---
    print("\n--- Generating 3D Risk Surface ---")
    
    # Scenario Simulator Input
    print("\n" + "-"*40)
    print("   SCENARIO SIMULATOR")
    sim_input = input("   Enter Scenario (e.g. -0.10): ").strip()
    current_yield_dev = float(sim_input) if sim_input else None

    baseline = comm_config.get('baseline_yield', 100)
    
    # We use the OLD Linear Regression R2 for display, or we could calc a new one.
    # For now, we pass a placeholder since this is a forward-looking model.
    dummy_r2 = 0.42 
    
    file_path = generate_interactive_surface(
        None, # Model not needed for viz
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
