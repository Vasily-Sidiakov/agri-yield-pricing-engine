from src.utils import load_config
from src.data_loader import fetch_market_data, fetch_weather_data
from src.stochastic import StochasticWeatherGenerator
from src.biophysics import BiophysicalTwin
from src.pricing import StochasticPricingEngine
from src.visualization import generate_interactive_surface, print_executive_summary
from src.models import train_yield_model, analyze_price_risk, calculate_price_sensitivity

import sys
import webbrowser
import os
import numpy as np
import pandas as pd
import warnings

# Silence numpy polyfit RankWarning that can pop up in small samples
try:
    warnings.simplefilter("ignore", np.RankWarning)
except AttributeError:
    from numpy.exceptions import RankWarning
    warnings.simplefilter("ignore", RankWarning)


def main():
    print("==========================================")
    print("   AGRI-YIELD PRICING ENGINE (v5.3)       ")
    print("   (Biophysical + Historical Coupling)    ")
    print("==========================================\n")

    # ------------------------------------------------------------------
    # 1. LOAD CONFIG AND SELECT COMMODITY
    # ------------------------------------------------------------------
    config = load_config()
    if not config:
        sys.exit("Failed to load configuration.")

    alias_map = {}
    print("Available Commodities:")
    for key, data in config["commodities"].items():
        print(f" - {data['name']}")
        simple_name = key.split("_")[0]
        alias_map[simple_name] = key
        alias_map[key] = key
        clean_name = data["name"].lower().split(" (")[0]
        alias_map[clean_name] = key

    print("\n" + "-" * 40)
    user_input = input("Which commodity would you like to analyze? ").strip().lower()

    if user_input in alias_map:
        selected_key = alias_map[user_input]
    else:
        print(f"\nSorry, I didn't recognize '{user_input}'.")
        sys.exit()

    comm_config = config["commodities"][selected_key]
    print(f"\n--- Initiating Analysis for {comm_config['name']} ---")

    ticker = comm_config["ticker"]

    # ------------------------------------------------------------------
    # 2. DATA INGESTION
    # ------------------------------------------------------------------
    print(f"   > Ingesting 44 Years of Satellite Data...")
    market_df = fetch_market_data(ticker, start_date="1980-01-01")
    current_price = float(market_df["price"].iloc[-1])

    weather_df = None
    for region in comm_config["regions"]:
        w_df = fetch_weather_data(
            region["latitude"],
            region["longitude"],
            "1980-01-01",
            "2024-12-31",
            region["name"],
        )
        if weather_df is None:
            weather_df = w_df
        else:
            weather_df = pd.concat([weather_df, w_df])

    # ------------------------------------------------------------------
    # 3. STOCHASTIC WEATHER GENERATION
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Stochastic Weather Simulation ---")
    weather_gen = StochasticWeatherGenerator(selected_key)

    weather_gen.calibrate(weather_df, variable="tmax")
    weather_gen.calibrate(weather_df, variable="vpd")

    print("   > Running Monte Carlo Simulation (10,000 Paths)...")
    tmax_paths = weather_gen.simulate("2024-01-01", days_ahead=180, variable="tmax")
    vpd_paths = weather_gen.simulate("2024-01-01", days_ahead=180, variable="vpd")
    rain_paths = weather_gen.simulate_rain(days_ahead=180, n_paths=10000)

    aad_paths = weather_gen.calculate_accumulated_stress(vpd_paths)

    # ------------------------------------------------------------------
    # 4. BIOPHYSICAL DIGITAL TWIN
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Biophysical Crop Modeling ---")
    print("   > Solving Differential Equations (Biomass/Soil Moisture)...")

    bio_engine = BiophysicalTwin(selected_key)

    # Twin returns a yield for each Monte Carlo path
    yield_paths, ks_history = bio_engine.solve_odes(
        tmax_paths, vpd_paths, days_ahead=180
    )

    baseline_simulated_yield = float(np.mean(yield_paths))
    if baseline_simulated_yield < 1e-6:
        baseline_simulated_yield = 1.0

    yield_deviations = (yield_paths - baseline_simulated_yield) / baseline_simulated_yield

    # ------------------------------------------------------------------
    # 4a. ALIGN SIMULATED YIELD VOLATILITY WITH HISTORY
    # ------------------------------------------------------------------
    hist_std = None
    features_path = f"data/processed/features_{selected_key}.csv"
    features_df = None
    if os.path.exists(features_path):
        try:
            features_df = pd.read_csv(features_path)
            if "yield_deviation" in features_df.columns:
                hist_std = float(features_df["yield_deviation"].std())
        except Exception as e:
            print(f"   > (Warning) Could not load historical features: {e}")

    sim_std = float(np.std(yield_deviations))
    if hist_std is not None and hist_std > 0 and sim_std > 1e-6:
        scale = hist_std / sim_std
        print(
            f"   > Rescaling simulated yield volatility "
            f"from {sim_std:.3f} to {hist_std:.3f} (scale={scale:.2f})."
        )
        yield_deviations = yield_deviations * scale
        # Reconstruct yield paths to be consistent with the rescaled deviations
        yield_paths = baseline_simulated_yield * (1.0 + yield_deviations)
    else:
        print("   > Using raw simulated yield volatility (no historical rescale).")

    # ------------------------------------------------------------------
    # 4b. HISTORICAL YIELD & PRICE RISK ANALYSIS
    # ------------------------------------------------------------------
    historical_beta = 0.0
    r2_score = 0.0
    risk_df = pd.DataFrame()

    if features_df is not None and not features_df.empty:
        try:
            yield_model, r2_score = train_yield_model(features_df)
        except Exception as e:
            print(f"   > (Warning) Yield model training failed: {e}")
            r2_score = 0.0
            yield_model = None

        try:
            risk_df = analyze_price_risk(features_df, ticker)
            historical_beta = calculate_price_sensitivity(risk_df)
        except Exception as e:
            print(f"   > (Warning) Price risk analysis failed: {e}")
            historical_beta = 0.0
            risk_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. FINANCIAL PRICING (MONTE CARLO)
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Pricing Engine ---")
    print(f"   > Pricing off current futures level: ${current_price:.2f}")

    pricing_engine = StochasticPricingEngine(current_price)
    biomass_proxy = np.outer(np.linspace(0, 1, 180), yield_paths)

    price_paths, convenience_yields = pricing_engine.simulate_pricing_paths(
        biomass_proxy, aad_paths, days_ahead=180
    )

    final_prices = price_paths[-1, :]
    returns_pct = (final_prices - current_price) / current_price * 100.0

    # ------------------------------------------------------------------
    # 5a. MONTE CARLO-IMPLIED ELASTICITY (SANITY CHECK ONLY)
    # ------------------------------------------------------------------
    mc_sensitivity = 0.0
    yield_std = float(np.std(yield_deviations))

    if yield_std < 1e-4:
        print("   > Note: Simulated yield variance is tiny. MC elasticity ~ 0.")
    else:
        try:
            mc_sensitivity = np.polyfit(yield_deviations, returns_pct, 1)[0]
        except Exception:
            mc_sensitivity = 0.0

    # Hard clip for MC beta so numerics cannot run away
    if abs(mc_sensitivity) > 50.0:
        print(
            f"   > MC-implied elasticity ({mc_sensitivity:.1f}%% per 1.0 dev) "
            "looks unstable. Clipping for safety."
        )
        mc_sensitivity = float(
            np.sign(mc_sensitivity) * 50.0
        )  # ±50% per 1.0 yield deviation

    # ------------------------------------------------------------------
    # 5b. CHOOSE FINAL ELASTICITY (FAVOR HISTORY WHEN AVAILABLE)
    # ------------------------------------------------------------------
    sensitivity = mc_sensitivity
    source = "Monte Carlo (simulated)"

    if historical_beta != 0.0:
        sensitivity = float(historical_beta)
        source = "Historical futures data"

    elasticity = abs(sensitivity) / 100.0
    relationship = "INVERSE" if sensitivity < 0 else "DIRECT"

    print("\n--- Elasticity Summary ---")
    print(f"   > Source of elasticity: {source}")
    print(
        f"   > Yield→Price slope: {sensitivity:+.1f}% price per +1.0 in yield deviation"
    )
    print(f"   > Implied Market Elasticity: {elasticity:.2f}x ({relationship})")

    # Optional: high-level narrative summary using historical risk table
    if not risk_df.empty:
        print_executive_summary(comm_config["name"], r2_score, sensitivity, risk_df)

    # ------------------------------------------------------------------
    # 6. VISUALIZATION
    # ------------------------------------------------------------------
    print("\n--- Generating 3D Risk Surface ---")
    print("\n" + "-" * 40)
    print("   SCENARIO SIMULATOR")
    sim_input = input("   Enter Scenario Yield Deviation (e.g. -0.10): ").strip()

    current_yield_dev = float(sim_input) if sim_input else None

    baseline_yield = comm_config.get("baseline_yield", 100)

    file_path = generate_interactive_surface(
        None, sensitivity, baseline_yield, comm_config["name"], current_yield_dev
    )

    full_path = "file://" + os.path.abspath(file_path)
    print(f"   > Opening {file_path} in browser...")
    webbrowser.open(full_path)

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
