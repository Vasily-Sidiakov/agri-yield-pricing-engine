import numpy as np
import pandas as pd

class StochasticPricingEngine:
    """
    Implements the Gibson-Schwartz Two-Factor Model coupled with 
    Seasonal Heston Stochastic Volatility.
    
    Links Biophysical State (Biomass) directly to Financial State (Convenience Yield).
    """
    
    def __init__(self, current_price, risk_free_rate=0.04):
        self.S0 = current_price
        self.r = risk_free_rate
        
        # Heston Parameters (Calibrated to typical Ag markets)
        self.kappa_v = 2.0   # Speed of volatility mean reversion
        self.theta_v = 0.04  # Long-run average variance (20% vol squared)
        self.xi_v = 0.3      # Volatility of volatility
        self.rho = -0.5      # Correlation between Price and Volatility (Leverage effect)

    def calculate_convenience_yield(self, biomass_paths):
        """
        Maps Physical Biomass to Financial Convenience Yield using the Theory of Storage.
        
        Formula: delta_t = delta_min + alpha * exp(-beta * Biomass_t)
        Logic: Scarcity (Low Biomass) -> High Convenience Yield (Backwardation)
        """
        # Normalize Biomass (0 to 1 scale relative to max potential)
        # Avoid division by zero
        max_b = np.max(biomass_paths) + 1e-6
        normalized_biomass = biomass_paths / max_b
        
        # Scarcity Parameters
        delta_min = -0.05 # Cost of carry (storage costs dominate in glut)
        alpha = 0.25      # Max panic premium (25% backwardation)
        beta = 5.0        # Sensitivity to scarcity
        
        # Calculate Delta
        delta_t = delta_min + alpha * np.exp(-beta * normalized_biomass)
        
        return delta_t

    def simulate_pricing_paths(self, biomass_paths, aad_paths, days_ahead):
        """
        Generates N Futures Price Paths using a Jump-Diffusion Heston Model
        driven by Biophysical inputs.
        
        dS_t = S_t * (r - delta_t) * dt + S_t * sqrt(v_t) * dW_s
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
        """
        n_paths = biomass_paths.shape[1]
        dt = 1/252 # Daily time step (Trading days)
        sqrt_dt = np.sqrt(dt)
        
        # Initialize Arrays
        S = np.zeros((days_ahead, n_paths))
        v = np.zeros((days_ahead, n_paths))
        
        S[0, :] = self.S0
        v[0, :] = self.theta_v # Start at long-run volatility
        
        # Pre-calculate Convenience Yields from Biomass
        # This links Phase 2 (Biology) to Phase 3 (Finance)
        delta_matrix = self.calculate_convenience_yield(biomass_paths)
        
        # Generate Correlated Brownian Motions
        # Price and Volatility are negatively correlated (rho)
        Z1 = np.random.normal(0, 1, (days_ahead, n_paths))
        Z2 = np.random.normal(0, 1, (days_ahead, n_paths))
        
        dW_s = Z1 * sqrt_dt
        dW_v = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * sqrt_dt
        
        for t in range(1, days_ahead):
            # Previous States
            S_prev = S[t-1, :]
            v_prev = np.maximum(v[t-1, :], 0) # Variance cannot be negative
            delta_prev = delta_matrix[t-1, :]
            
            # --- WEATHER VOLATILITY AMPLIFIER ---
            # If Accumulated Atmospheric Demand (AAD) is rising fast, boost vol.
            # We look at the change in AAD from the weather engine
            aad_change = aad_paths[t, :] - aad_paths[t-1, :]
            stress_multiplier = 1.0 + (aad_change * 0.5) # Tuning parameter
            
            # Effective Volatility (Heston + Weather Stress)
            sigma_t = np.sqrt(v_prev) * stress_multiplier
            
            # --- 1. Heston Variance Process (dv_t) ---
            dv = self.kappa_v * (self.theta_v - v_prev) * dt + \
                 self.xi_v * np.sqrt(v_prev) * dW_v[t, :]
            
            v_new = v_prev + dv
            v[t, :] = np.maximum(v_new, 1e-6) # Floor variance
            
            # --- 2. Gibson-Schwartz Price Process (dS_t) ---
            # Drift = Risk Free Rate - Convenience Yield
            drift = (self.r - delta_prev) * dt
            
            # Diffusion = Price * Volatility * Noise
            diffusion = sigma_t * dW_s[t, :]
            
            # Update Price (Geometric Brownian Motion dynamics)
            # S_new = S_prev * exp(drift - 0.5*sigma^2*dt + diffusion)
            # (Using Ito's Lemma formulation for stability)
            S_new = S_prev * np.exp(drift - 0.5 * sigma_t**2 * dt + diffusion)
            
            S[t, :] = S_new
            
        return S, delta_matrix
