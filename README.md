# AgriYield Pricing Engine (v3.3)

A Python-based quantitative research engine that models the non-linear relationship between agronomic stress (Vapor Pressure Deficit, Soil Moisture) and commodity futures returns.

## 📊 Project Overview
This tool automates the pricing of supply-side agricultural risk. It ingests 44 years of daily satellite weather data, models crop physiology, and calculates the "Price Elasticity" of yield shocks using a volatility surface.

**Key Capabilities:**
* **Global Scope:** Supports Corn, Soybeans, Wheat, Coffee, Cocoa, Sugar, Cotton, and Rice.
* **Advanced Agronomy:** Moves beyond simple rainfall/temperature to model **Vapor Pressure Deficit (VPD)** and **Root-Zone Soil Moisture**.
* **3D Visualization:** Generates interactive "Liquid Neon" volatility surfaces using Plotly WebGL.
* **Scenario Simulator:** Allows users to inject hypothetical yield shocks (e.g., "-10% Yield") to project specific price returns based on historical elasticity.

## 🚀 How It Works
1.  **Ingestion:** Fetches OHLC Futures data (Yahoo Finance) and Daily Weather (Open-Meteo) from 1980–2024. Handles API rate-limiting and caching automatically.
2.  **Processing:** Detrends historical yields to isolate weather-driven variance from technological trend growth.
3.  **Modeling:** Trains a Linear Regression model to predict Yield Deviation based on abiotic stress factors.
4.  **Pricing:** Calculates the **Price Elasticity (Beta)** of the commodity and maps it onto a 3D Volatility Surface ($\sigma\sqrt{t}$).

## 💡 Key Findings (Case Study: US Corn)
Running the engine on US Corn (1980-2024) revealed significant market asymmetry:
* **Market Elasticity:** **1.34x (Inverse)**.
* **Interpretation:** A 1% drop in yield typically drives a **1.34% rally** in prices.
* **Risk Asymmetry:** The market punishes oversupply (-21% return) significantly harder than it rewards shortages (+8.6% return) during the growing season window.

## 🛠 Tech Stack
* **Core:** Python 3.11+
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly (3D Interactive), Seaborn
* **Data Sources:** Open-Meteo API, Yahoo Finance API

## 💻 Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Vasily-Sidiakov/agri-yield-pricing-engine.git

## Navigate to the folder and install dependencies:
2. cd agri-yield-pricing-engine
pip install -r requirements.txt 

## Run the application
3. python main.py

4. Follow the interactive menu: Select a commodity (e.g., Corn), wait for the data ingestion, and enter a hypothetical yield deviation (e.g., -0.10) to visualize the risk surface.
