import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class StochasticWeatherGenerator:
    """
    Implements the Ornstein-Uhlenbeck (OU) process to simulate mean-reverting
    weather variables (Temperature, VPD).
    
    Mathematical Foundation:
    dX_t = kappa * (theta(t) - X_t) * dt + sigma * dW_t
    
    Where:
    - kappa: Mean reversion speed (How fast anomalies dissipate)
    - theta(t): Seasonal mean function (Sinusoidal)
    - sigma: Volatility of the weather
    - dW_t: Wiener process (Brownian motion)
    """
    
    def __init__(self, region_name):
        self.region_name = region_name
        self.params = {} # Stores kappa, sigma, theta parameters
        
    def _seasonal_mean_func(self, t, a0, a1, a2, a3, a4):
        """
        Models the seasonal cycle theta(t) using a 2-term Fourier Series.
        theta(t) = a0 + a1*sin(wt) + a2*cos(wt) + ...
        This captures the annual warming/cooling cycle.
        """
        w = 2 * np.pi / 365.25 # Annual frequency
        return (a0 + 
                a1 * np.sin(w * t) + a2 * np.cos(w * t) + 
                a3 * np.sin(2 * w * t) + a4 * np.cos(2 * w * t))

    def calibrate(self, historical_df, variable='tmax'):
        """
        Calibrates the OU process parameters (kappa, sigma) from historical data.
        
        Methodology:
        1. Fit the deterministic seasonal trend theta(t).
        2. Subtract theta(t) to get residuals (Anomalies).
        3. Regress X(t+1) against X(t) to find Mean Reversion Speed.
        """
        print(f"   > Calibrating Stochastic Model for {variable}...")
        
        # 1. Prepare Time Series
        df = historical_df.copy()
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Convert date to a continuous time axis for the math
        # We assume t=0 is the start of the dataset
        df['t'] = (df['date'] - df['date'].min()).dt.days
        
        Y = df[variable].values
        t = df['t'].values
        
        # 2. Fit Seasonality (Theta)
        # We use curve_fit to find the coefficients (a0, a1, ...) that best fit the history
        p0 = [np.mean(Y), -10, -10, 0, 0] # Initial guess
        try:
            popt, _ = curve_fit(self._seasonal_mean_func, t, Y, p0=p0)
            self.params[f'{variable}_seasonal'] = popt
        except RuntimeError:
            # Fallback if optimization fails (rare)
            print(f"     ! Warning: Seasonal fit failed for {variable}. Using simple mean.")
            self.params[f'{variable}_seasonal'] = [np.mean(Y), 0, 0, 0, 0]
            popt = self.params[f'{variable}_seasonal']

        # Calculate the deterministic mean for every day
        theta_t = self._seasonal_mean_func(t, *popt)
        
        # 3. Calculate Residuals (The Anomaly)
        # X_t is how far today's temp is from the "Normal" temp
        X_t = Y - theta_t
        
        # 4. Estimate Mean Reversion (Kappa) and Volatility (Sigma)
        # We use the discrete time solution of the OU process:
        # X_{t+1} = X_t * exp(-kappa * dt) + noise
        # This is a linear regression: Y = slope * X
        
        X_current = X_t[:-1] # Today's anomaly
        X_next = X_t[1:]     # Tomorrow's anomaly
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_current.reshape(-1, 1), X_next)
        
        slope = reg.coef_[0]
        
        # Recover Kappa from the slope
        # slope = exp(-kappa * dt), where dt = 1 day
        # kappa = -ln(slope)
        if slope > 0:
            kappa = -np.log(slope)
        else:
            kappa = 0.5 # Fallback for noisy data
        
        # Calculate Residuals of the regression to find Sigma
        # The standard deviation of these residuals is the daily volatility
        residuals = X_next - (X_current * slope)
        sigma = np.std(residuals) * np.sqrt(2 * kappa / (1 - slope**2))
        
        self.params[f'{variable}_kappa'] = kappa
        self.params[f'{variable}_sigma'] = sigma
        
        print(f"     -> {variable.upper()} Params: Reversion Speed (k)={kappa:.3f}, Volatility (s)={sigma:.2f}")

    def simulate(self, start_date, days_ahead=180, n_paths=1000, variable='tmax'):
        """
        Generates N Monte Carlo paths for the specified weather variable
        using the Euler-Maruyama discretization scheme.
        
        dX = kappa * (0 - X) * dt + sigma * dW
        Total Value = Seasonal_Mean(t) + X_t
        """
        if f'{variable}_kappa' not in self.params:
            print(f"   ! Error: Model for {variable} not calibrated.")
            return None

        # Retrieve parameters
        kappa = self.params[f'{variable}_kappa']
        sigma = self.params[f'{variable}_sigma']
        popt = self.params[f'{variable}_seasonal']
        
        # Time setup
        # We need continuous time t for the seasonality function
        # We approximate t starting from day_of_year of start_date
        start_doy = pd.to_datetime(start_date).dayofyear
        t_axis = np.arange(start_doy, start_doy + days_ahead)
        
        # 1. Calculate Deterministic Seasonality for the whole horizon
        # Shape: (days_ahead,)
        theta_t = self._seasonal_mean_func(t_axis, *popt)
        
        # 2. Initialize Paths
        # X represents the anomaly (deviation from mean)
        # We start at X=0 (assuming we start from a 'normal' day, or we could pass last val)
        X = np.zeros((days_ahead, n_paths))
        
        # 3. Euler-Maruyama Simulation Loop
        dt = 1.0 # Daily time step
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, days_ahead):
            # Previous state
            x_prev = X[i-1, :]
            
            # Stochastic Shock (Wiener Process)
            dW = np.random.normal(0, 1, size=n_paths) * sqrt_dt
            
            # OU Process Update: Mean reversion to 0 + Random Shock
            dx = -kappa * x_prev * dt + sigma * dW
            
            # Update state
            X[i, :] = x_prev + dx
            
        # 4. Reconstruct Total Value (Seasonality + Anomaly)
        # We broadcast theta_t (days, 1) across the paths (days, paths)
        simulated_paths = X + theta_t[:, np.newaxis]
        
        return simulated_paths

    def calculate_accumulated_stress(self, vpd_paths, threshold=1.5):
        """
        Calculates the Accumulated Atmospheric Demand (AAD) integral.
        
        Integral_0^T max(VPD_t - Threshold, 0) dt
        
        This represents the 'Memory' of stress. A high integral means the 
        plant has been thirsting for a long time.
        """
        # 1. Identify Stress Days (Where VPD > Threshold)
        # vpd_paths shape: (days, paths)
        stress_event = np.maximum(vpd_paths - threshold, 0)
        
        # 2. Integrate over time (Cumulative Sum)
        aad_paths = np.cumsum(stress_event, axis=0)
        
        return aad_paths
