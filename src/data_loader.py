import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import yfinance as yf
import os
import time
from src.utils import LoadingSpinner

# --- 1. Setup for Weather API ---
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def fetch_market_data(ticker, start_date="1980-01-01"):
    """
    Downloads historical futures data from Yahoo Finance.
    CACHING ENABLED: Checks for local CSV before hitting API.
    """
    output_path = f"data/raw/market_{ticker}.csv"
    
    # --- CACHE CHECK ---
    if os.path.exists(output_path):
        print(f"   > [CACHE HIT] Loading Market Data for {ticker} from disk...")
        # Read and ensure index is Datetime
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        return df
    
    # --- DOWNLOAD IF MISSING ---
    with LoadingSpinner(f"> Downloading Market Data for {ticker}"):
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
    
    df.to_csv(output_path)
    print(f"   > Market Data Downloaded: {len(df)} records")
    return df

def fetch_weather_data(lat, lon, start_date, end_date, region_name):
    """
    Downloads historical daily weather.
    CACHING ENABLED: Skips API calls if data exists locally.
    """
    output_path = f"data/raw/weather_{region_name}.csv"
    
    # --- CACHE CHECK ---
    if os.path.exists(output_path):
        print(f"   > [CACHE HIT] Loading Weather Data for {region_name} from disk...")
        # Crucial: Parse 'date' column so .dt accessor works later
        df = pd.read_csv(output_path, parse_dates=['date'])
        return df

    # --- DOWNLOAD IF MISSING ---
    print(f"   > [CACHE MISS] Downloading Weather for {region_name} (This may take time)...")

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    
    all_chunks = []
    
    for chunk_start in range(start_year, end_year + 1, 10):
        chunk_end = min(chunk_start + 9, end_year)
        c_start_date = f"{chunk_start}-01-01"
        c_end_date = f"{chunk_end}-12-31"
        if chunk_end >= 2025: c_end_date = "2024-12-31"

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
        
        msg = f"Acquiring Satellite Data ({chunk_start}-{chunk_end})"
        with LoadingSpinner(msg):
            try:
                time.sleep(6) # Rate limiting
                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]

                daily = response.Daily()
                
                # Extract variables
                vars_dict = {
                    "tmax": daily.Variables(0).ValuesAsNumpy(),
                    "tmin": daily.Variables(1).ValuesAsNumpy(),
                    "precip": daily.Variables(2).ValuesAsNumpy(),
                    "vpd": daily.Variables(3).ValuesAsNumpy(),
                    "soil_moist": daily.Variables(4).ValuesAsNumpy()
                }

                date_range = pd.date_range(
                    start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                    end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                    freq = pd.Timedelta(seconds = daily.Interval()),
                    inclusive = "left"
                )

                chunk_df = pd.DataFrame(data = {"date": date_range, **vars_dict})
                all_chunks.append(chunk_df)
                
            except Exception as e:
                time.sleep(10)
                continue

    if not all_chunks:
        print(f"   > Error: No weather data could be retrieved for {region_name}")
        return None

    full_df = pd.concat(all_chunks)
    full_df = full_df.sort_values('date')

    full_df.to_csv(output_path, index=False)
    print(f"   > Download Complete: {len(full_df)} days cached to {output_path}")
    return full_df
