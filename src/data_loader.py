import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import yfinance as yf
import os
import time
from src.utils import LoadingSpinner  # <--- NEW IMPORT

# --- 1. Setup for Weather API ---
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def fetch_market_data(ticker, start_date="1980-01-01"):
    """
    Downloads historical futures data from Yahoo Finance.
    """
    # Use the spinner for the market data too
    with LoadingSpinner(f"> Fetching Market Data for {ticker}"):
        try:
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False, multi_level_index=False)
        except TypeError:
            df = yf.download(ticker, start=start_date, progress=False)

    if df.empty:
        print(f"   > Warning: No data found for {ticker}. Check ticker symbol.")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' in df.columns:
        df = df[['Close']]
    elif 'Adj Close' in df.columns:
        df = df[['Adj Close']]
    else:
        df = df.iloc[:, 0]
        
    df.columns = ['price']
    
    output_path = f"data/raw/market_{ticker}.csv"
    df.to_csv(output_path)
    print(f"   > Market Data Acquired: {len(df)} records")
    return df

def fetch_weather_data(lat, lon, start_date, end_date, region_name):
    """
    Downloads historical daily weather in 10-year chunks with DELAYS.
    Includes Advanced Agronomy Metrics: VPD and Soil Moisture.
    """
    output_path = f"data/raw/weather_{region_name}.csv"
    
    print(f"   > Target Region: {region_name}")

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    
    all_chunks = []
    
    # Loop from start_year to end_year in steps of 10
    for chunk_start in range(start_year, end_year + 1, 10):
        chunk_end = min(chunk_start + 9, end_year)
        
        c_start_date = f"{chunk_start}-01-01"
        c_end_date = f"{chunk_end}-12-31"
        
        if chunk_end >= 2025: 
             c_end_date = "2024-12-31"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": c_start_date,
            "end_date": c_end_date,
            "daily": [
                "temperature_2m_max", 
                "temperature_2m_min", 
                "precipitation_sum",
                "vapor_pressure_deficit_max",      
                "soil_moisture_28_to_100cm_mean"   
            ],
            "timezone": "auto"
        }
        
        # === THE NEW ANIMATION ===
        # We wrap the pause AND the download in the spinner
        msg = f"Acquiring Satellite Data ({chunk_start}-{chunk_end})"
        with LoadingSpinner(msg):
            try:
                # Sleep first (rate limiting)
                time.sleep(6) 
                
                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]

                daily = response.Daily()
                daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
                daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
                daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
                daily_vpd = daily.Variables(3).ValuesAsNumpy()
                daily_soil = daily.Variables(4).ValuesAsNumpy()

                date_range = pd.date_range(
                    start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                    end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                    freq = pd.Timedelta(seconds = daily.Interval()),
                    inclusive = "left"
                )

                chunk_df = pd.DataFrame(data = {
                    "date": date_range,
                    "tmax": daily_temperature_2m_max,
                    "tmin": daily_temperature_2m_min,
                    "precip": daily_precipitation_sum,
                    "vpd": daily_vpd,          
                    "soil_moist": daily_soil   
                })
                
                all_chunks.append(chunk_df)
                
            except Exception as e:
                # If error, simply pass so the loop continues (cleaner UI)
                # In a pro app, you might log this to a file
                time.sleep(10)
                continue

    if not all_chunks:
        print(f"   > Error: No weather data could be retrieved for {region_name}")
        return None

    full_df = pd.concat(all_chunks)
    full_df = full_df.sort_values('date')

    full_df.to_csv(output_path, index=False)
    # Final confirmation message
    print(f"   > Weather History Downloaded: {len(full_df)} days of data")
    return full_df