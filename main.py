from src.utils import load_config
from src.data_loader import fetch_market_data, fetch_weather_data
from src.features import process_weather_features
from src.models import train_yield_model, analyze_price_risk, calculate_price_sensitivity
from src.visualization import generate_interactive_surface, print_executive_summary
import sys
import webbrowser
import os

def main():
    print("==========================================")
    print("   AGRI-YIELD PRICING ENGINE (v3.3)       ")
    print("   (Professional Edition)                 ")
    print("==========================================")
    
    # 1. Load Configuration
    config = load_config()
    if not config:
        sys.exit("Failed to load configuration.")
    
    # 2. Build Smart Menu & Alias Map
    alias_map = {}
    print("\nAvailable Commodities:")
    for key, data in config['commodities'].items():
        print(f" - {data['name']}")
        simple_name = key.split('_')[0] 
        alias_map[simple_name] = key
        alias_map[key] = key
        clean_name = data['name'].lower().split(' (')[0]
        alias_map[clean_name] = key

    # 3. Input
    print("\n" + "-"*40)
    user_input = input("Which commodity would you like to analyze? ").strip().lower()
    
    if user_input in alias_map:
        selected_key = alias_map[user_input]
    else:
        print(f"\nSorry, I didn't recognize '{user_input}'.")
        sys.exit()
        
    comm_config = config['commodities'][selected_key]
    print(f"\n--- Initiating Analysis for {comm_config['name']} ---")

    # 4. Data Ingestion
    print(f"   > Checking Market Data for {comm_config['ticker']}...")
    fetch_market_data(comm_config['ticker'], start_date="1980-01-01")
    
    print(f"   > Checking Weather Data...")
    for region in comm_config['regions']:
        fetch_weather_data(
            lat=region['latitude'], 
            lon=region['longitude'], 
            start_date="1980-01-01", 
            end_date="2024-12-31", 
            region_name=region['name']
        )

    # 5. Feature Engineering
    print("\n--- Processing Agronomic Features ---")
    features_df = process_weather_features(selected_key)
    
    if features_df is None:
        sys.exit("Critical Error: Feature processing failed.")

    # 6. Modeling
    print("\n--- Training Yield Model ---")
    model, r2_score = train_yield_model(features_df)
    print(f"   > Yield Model Accuracy (R2): {r2_score:.2f}")

    # 7. Risk Analysis
    print("\n--- Generating Risk Metrics ---")
    risk_df = analyze_price_risk(features_df, comm_config['ticker'])
    
    if risk_df.empty:
        print("Warning: Not enough price data aligned with harvest years.")
    else:
        summary = risk_df.groupby('yield_bucket')['harvest_return_pct'].describe()[['count', 'mean']]
        print(summary)

        # 8. Sensitivity
        sensitivity = calculate_price_sensitivity(risk_df)
        elasticity = abs(sensitivity) / 100
        relationship = "INVERSE" if sensitivity < 0 else "DIRECT"
        
        print(f"\n   > Market Elasticity: {elasticity:.2f}x ({relationship})")
        print(f"     (Interpretation: A 1% Yield move drives a {elasticity:.2f}% Price change)")
        
        # 9. SCENARIO SIMULATION (Meeting the Prompt Requirement)
        print("\n" + "-"*40)
        print("   SCENARIO SIMULATOR")
        print("   Enter a hypothetical yield deviation to see where it lands on the risk surface.")
        print("   (e.g., -0.05 for a 5% crop failure, or 0.10 for a bumper crop)")
        sim_input = input("   Enter Scenario (or press Enter to skip): ").strip()
        
        current_yield_dev = None
        if sim_input:
            try:
                current_yield_dev = float(sim_input)
                print(f"   > Simulating Yield Deviation: {current_yield_dev:.1%}")
            except ValueError:
                print("   > Invalid number. Skipping simulation.")

        # 10. Visualization
        print("\n--- Generating 3D Risk Surface ---")
        baseline = comm_config.get('baseline_yield', 100)
        
        file_path = generate_interactive_surface(
            model, 
            sensitivity, 
            baseline, 
            comm_config['name'],
            current_yield_dev=current_yield_dev # Pass the scenario
        )
        
        full_path = "file://" + os.path.abspath(file_path)
        print(f"   > Opening {file_path} in browser...")
        webbrowser.open(full_path)
        
        # 11. Summary
        print("\n" + "-"*40)
        user_summary = input("Would you like a summarized report of these findings? (y/n): ").strip().lower()
        
        if user_summary == 'y':
            print_executive_summary(comm_config['name'], r2_score, sensitivity, risk_df)

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()