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
Prerequisite: You need Python
Before starting, the user must have Python installed.

Check: Open Terminal/Command Prompt and type python --version.

If missing: Download it from python.org.

Step 1: Download the Project
We need to get the files from the internet onto your computer.

Open your Terminal (Mac) or Command Prompt (Windows).

Paste this command and hit Enter: git clone https://github.com/Vasily-Sidiakov/agri-yield-pricing-engine.git

## Navigate to the folder and install dependencies:
Step 2: Go Inside the Folder
Right now, your terminal is looking at your main user folder. You need to step inside the project folder you just downloaded.

Paste this command: cd agri-yield-pricing-engine

## Run the application
Step 3: Create the "Brain" (Virtual Environment)
We need to create an isolated box so this project's tools don't mess up your computer. This is called a "Virtual Environment" (venv).

If you are on Mac, paste this command: python3 -m venv venv
source venv/bin/activate

If you are on Windows, paste this command: python -m venv venv
venv\Scripts\activate

Checkpoint: You should now see (venv) at the very start of your terminal line. This means the "Brain" is turned on.

Step 4: Install the Tools
Now we need to teach the brain how to do math (Pandas) and draw 3D charts (Plotly). We do this by reading the recipe list (requirements.txt).

Paste this command: pip install -r requirements.txt

Step 5: Run the Engine
Once everything has been installed, you are ready to run the program. To do so,

Paste this command: python main.py

How to Use It (Once it's running)
Select a Crop: When the menu appears, type Corn and hit Enter.

Wait: The program will pause while it downloads 44 years of satellite data. You will see a spinner animation.

Simulate: It will ask for a scenario. Type a value such as -0.10, which indicates a 10% decrease in crop yield. A value such as 0.05 indicates a 5% increase in crop yield.

View: Your web browser will automatically open with the 3D Liquid Neon Volatility Surface
