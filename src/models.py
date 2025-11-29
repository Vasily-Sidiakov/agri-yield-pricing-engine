import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_yield_model(features_df: pd.DataFrame):
    """
    Train a Ridge regression model to explain yield_deviation from weather / stress features.

    Returns
    -------
    model : sklearn Pipeline
        StandardScaler + Ridge(alpha=1.0)
    r2 : float
        In-sample R² of the fit. If data is insufficient, returns 0.0.
    """
    # Target column must exist
    if "yield_deviation" not in features_df.columns:
        print("   > (Warning) 'yield_deviation' not found in features_df.")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model, 0.0

    # Drop rows without yield deviation
    data = features_df.dropna(subset=["yield_deviation"])
    if data.empty:
        print("   > (Warning) No rows with non-null yield_deviation.")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model, 0.0

    # ------------------------------------------------------------
    # 1. Feature set (aligned with src/features.py)
    # ------------------------------------------------------------
    potential_features = [
        # Base growth / water balance
        "weighted_bio_growth",
        "weighted_precip",

        # Soil moisture & combined stress
        "weighted_soil_moist",
        "stress_x_dryness",      # Heat * dryness interaction
        "precip_sq",             # Flood risk via convexity

        # Accumulated weather stress (frost/heat degree days)
        "weighted_acc_stress",

        # Memory term: previous-year yield deviation (for perennial / biennial crops)
        "lag_1_yield_dev",
    ]

    # Only keep columns that actually exist in the data
    X_cols = [c for c in potential_features if c in data.columns]
    if not X_cols:
        print("   > (Warning) No usable feature columns found.")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model, 0.0

    # Drop rows that have NaNs in the selected feature columns
    model_data = data.dropna(subset=X_cols)
    if len(model_data) < 5:
        print("   > (Warning) Not enough data depth for advanced modeling.")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model, 0.0

    X = model_data[X_cols]
    y = model_data["yield_deviation"]

    # ------------------------------------------------------------
    # 2. Train Ridge regression (L2 regularization)
    # ------------------------------------------------------------
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    r2 = float(model.score(X, y))

    return model, r2


def analyze_price_risk(features_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    For each crop year, link realized yield deviation to realized futures returns
    over the harvest window (roughly May–October in this simplified version).

    Parameters
    ----------
    features_df : DataFrame
        Must contain at least 'year' and 'yield_deviation'.
    ticker : str
        Ticker symbol used for market data, e.g. 'ZC=F', 'ZS=F', etc.

    Returns
    -------
    DataFrame with columns:
        - year
        - yield_bucket (text description: low / normal / high yield)
        - yield_deviation (float)
        - harvest_return_pct (float, % return over the window)
    """
    # Make sure we have the necessary columns
    if "year" not in features_df.columns or "yield_deviation" not in features_df.columns:
        print("   > (Warning) features_df must contain 'year' and 'yield_deviation'.")
        return pd.DataFrame()

    # Load stored futures prices
    try:
        price_path = f"data/raw/market_{ticker}.csv"
        price_df = pd.read_csv(price_path)

        if "Date" in price_df.columns:
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.set_index("Date", inplace=True)
        else:
            # Fallback if index already holds dates
            price_df.index = pd.to_datetime(price_df.index)

        if "price" not in price_df.columns:
            raise ValueError("price column not found in market data")

    except Exception as e:
        print(f"   > (Warning) Could not load market data for {ticker}: {e}")
        return pd.DataFrame()

    results = []

    # Iterate over years with valid yield deviations
    for _, row in features_df.dropna(subset=["yield_deviation"]).iterrows():
        try:
            year = int(row["year"])
            deviation = float(row["yield_deviation"])
        except Exception:
            continue

        # Simple bucketing for narrative
        if deviation < -0.05:
            bucket = "Low Yield (Bullish)"
        elif deviation > 0.05:
            bucket = "High Yield (Bearish)"
        else:
            bucket = "Normal Yield"

        try:
            # Harvest window: roughly May–Oct
            # (In production you’d want crop-specific windows)
            start_date = f"{year}-05-01"
            end_date = f"{year}-10-01"

            window = price_df.loc[start_date:end_date]
            if len(window) < 10:
                # Not enough price data for this year
                continue

            p_start = float(window.iloc[0]["price"])
            p_end = float(window.iloc[-1]["price"])
            if p_start <= 0:
                continue

            pct_return = (p_end - p_start) / p_start * 100.0

            results.append(
                {
                    "year": year,
                    "yield_bucket": bucket,
                    "yield_deviation": deviation,
                    "harvest_return_pct": pct_return,
                }
            )
        except Exception:
            # Skip year if anything goes wrong for that window
            continue

    return pd.DataFrame(results)


def calculate_price_sensitivity(risk_df: pd.DataFrame) -> float:
    """
    Compute the historical slope (beta) of price returns vs yield deviations.

    Parameters
    ----------
    risk_df : DataFrame
        Must contain 'yield_deviation' and 'harvest_return_pct'.

    Returns
    -------
    float
        Slope of regression line: % price move per +1.0 in yield_deviation.
        If insufficient data, returns 0.0.
    """
    if risk_df is None or risk_df.empty:
        return 0.0

    if "yield_deviation" not in risk_df.columns or "harvest_return_pct" not in risk_df.columns:
        return 0.0

    df = risk_df.dropna(subset=["yield_deviation", "harvest_return_pct"])
    if len(df) < 5:
        return 0.0

    # Simple OLS via polyfit (degree 1)
    slope, _ = np.polyfit(df["yield_deviation"], df["harvest_return_pct"], 1)
    return float(slope)
