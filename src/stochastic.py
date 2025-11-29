import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class StochasticWeatherGenerator:
    """
    Implements the Ornstein-Uhlenbeck (OU) process to simulate mean-reverting
    weather variables (Temperature, VPD) and a Poisson process for Rainfall.
    """
    
    def __init__(self, region_name):
        self.region_name = region_name
        self.params = {} 
        
    def _seasonal_mean_func(self, t, a0, a1, a2, a3, a4):
        w = 2 * np.pi / 365.25 
        return (a0 + 
                a1 * np.sin(w * t) + a2 * np.cos(w * t) + 
                a3 * np.sin(2 * w * t) + a4 * np.cos(2 * w * t))

    def calibrate(self, historical_df, variable='tmax'):
        """
        Calibrates the OU process parameters (kappa, sigma) from historical data.
        """
        # Safety check for missing columns (e.g. if calibrating rain)
        if variable not in historical_df.columns:
            if variable == 'precip': return # Rain is handled differently
            print(f"   ! Warning: {variable} not found in history.")
            return

        print(f"   > Calibrating Stochastic Model for {variable}...")
        
        df = historical_df.copy()
        df['t'] = (df['date'] - df['date'].min()).dt.days
        
        Y = df[variable].values
        t = df['t'].values
        
        # Fit Seasonality
        p0 = [np.mean(Y), -10, -10, 0, 0]
        try:
            popt, _ = curve_fit(self._seasonal_mean_func, t, Y, p0=p0)
            self.params[f'{variable}_seasonal'] = popt
        except RuntimeError:
            self.params[f'{variable}_seasonal'] = [np.mean(Y), 0, 0, 0, 0]
            popt = self.params[f'{variable}_seasonal']

        theta_t = self._seasonal_mean_func(t, *popt)
        X_t = Y - theta_t
        
        # Estimate Kappa/Sigma
        X_current = X_t[:-1]
        X_next = X_t[1:]
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_current.reshape(-1, 1), X_next)
        
        slope = reg.coef_[0]
        if slope > 0: kappa = -np.log(slope)
        else: kappa = 0.5 
        
        residuals = X_next - (X_current * slope)
        sigma = np.std(residuals) * np.sqrt(2 * kappa / (1 - slope**2))
        
        self.params[f'{variable}_kappa'] = kappa
        self.params[f'{variable}_sigma'] = sigma
        
        print(f"     -> {variable.upper()} Params: Reversion Speed (k)={kappa:.3f}, Volatility (s)={sigma:.2f}")

    def simulate(self, start_date, days_ahead=180, n_paths=1000, variable='tmax'):
        """
        Generates N Monte Carlo paths using Euler-Maruyama.
        """
        if f'{variable}_kappa' not in self.params:
            # Fallback for Tmin if not explicitly calibrated (Estimate from Tmax)
            if variable == 'tmin' and 'tmax_seasonal' in self.params:
                # Assume Tmin follows Tmax pattern but 12 degrees cooler
                tmax_paths = self.simulate(start_date, days_ahead, n_paths, 'tmax')
                noise = np.random.normal(0, 2.0, size=tmax_paths.shape) # Add noise
                return tmax_paths - 12.0 + noise
            return np.zeros((days_ahead, n_paths))

        kappa = self.params[f'{variable}_kappa']
        sigma = self.params[f'{variable}_sigma']
        popt = self.params[f'{variable}_seasonal']
        
        start_doy = pd.to_datetime(start_date).dayofyear
        t_axis = np.arange(start_doy, start_doy + days_ahead)
        
        theta_t = self._seasonal_mean_func(t_axis, *popt)
        
        X = np.zeros((days_ahead, n_paths))
        dt = 1.0
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, days_ahead):
            x_prev = X[i-1, :]
            dW = np.random.normal(0, 1, size=n_paths) * sqrt_dt
            dx = -kappa * x_prev * dt + sigma * dW
            X[i, :] = x_prev + dx
            
        simulated_paths = X + theta_t[:, np.newaxis]
        return simulated_paths

    def simulate_rain(self, days_ahead=180, n_paths=1000):
        """
        Simulates Rainfall using a Marked Poisson Process.
        Rain arrives randomly (Poisson) and amount is Exponential.
        """
        # Probability of rain on any given day (approx 15%)
        lambda_rain = 0.15 
        
        # Intensity of rain (Mean mm per event)
        mu_rain = 15.0
        
        is_raining = np.random.rand(days_ahead, n_paths) < lambda_rain
        rain_amounts = np.random.exponential(mu_rain, size=(days_ahead, n_paths))
        
        return rain_amounts * is_raining

    def calculate_accumulated_stress(self, vpd_paths, threshold=1.5):
        stress_event = np.maximum(vpd_paths - threshold, 0)
        aad_paths = np.cumsum(stress_event, axis=0)
        return aad_paths
