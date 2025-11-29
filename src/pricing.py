import numpy as np
import pandas as pd

class StochasticPricingEngine:
    """
    Implements a Structural Scarcity Model coupled with Seasonal Heston Volatility.
    
    Logic:
    Price is driven fundamentally by the Supply/Demand ratio (Biomass).
    Volatility is driven by the rate of change in Weather Stress.
    """
    
    def __init__(self, current_price, risk_free_rate=0.04):
        self.S0 = current_price
        self.r = risk_free_rate
        
        # Heston Parameters
        self.kappa_v = 3.0   
        self.theta_v = 0.09  
        self.xi_v = 0.6      
        self.rho = -0.7      

    def simulate_pricing_paths(self, biomass_paths, aad_paths, days_ahead):
        """
        Generates Futures Price Paths based on Structural Scarcity.
        
        Formula:
        Fundamental_Price = S0 * (Baseline_Biomass / Current_Biomass) ^ Scarcity_Factor
        Final_Price = Fundamental_Price * Heston_Noise
        """
        n_paths = biomass_paths.shape[1]
        dt = 1/252 
        sqrt_dt = np.sqrt(dt)
        
        # Initialize Arrays
        S = np.zeros((days_ahead, n_paths))
        v = np.zeros((days_ahead, n_paths))
        
        S[0, :] = self.S0
        v[0, :] = self.theta_v 
        
        # 1. Establish Baseline (Expected) Biomass
        # The market expects the "Average" outcome. Deviations from this drive price.
        # We take the mean of the biomass paths at each time step as the "Market Consensus"
        market_consensus_biomass = np.mean(biomass_paths, axis=1)
        
        # Scarcity Factor (Lambda)
        # How violently price reacts to a 1% miss in supply.
        # 2.5 means a 1% drop in supply = 2.5% increase in price.
        scarcity_factor = 2.5 
        
        # Correlated Brownian Motions for Volatility
        Z1 = np.random.normal(0, 1, (days_ahead, n_paths))
        Z2 = np.random.normal(0, 1, (days_ahead, n_paths))
        
        dW_v = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * sqrt_dt
        
        # Noise accumulator (Random Walk for price noise)
        noise_accum = np.ones((days_ahead, n_paths))
        
        for t in range(1, days_ahead):
            # --- A. Heston Volatility Process ---
            v_prev = np.maximum(v[t-1, :], 1e-6)
            
            # Weather Panic Multiplier
            aad_change = aad_paths[t, :] - aad_paths[t-1, :]
            stress_multiplier = 1.0 + (aad_change * 2.0)
            
            dv = self.kappa_v * (self.theta_v - v_prev) * dt + \
                 self.xi_v * np.sqrt(v_prev) * dW_v[t, :]
            
            v_new = np.maximum(v_prev + dv, 1e-6)
            v[t, :] = v_new
            
            # Effective Volatility for this step
            sigma_t = np.sqrt(v_new) * stress_multiplier
            
            # --- B. Fundamental Price (The Structural Driver) ---
            # Ratio: Expected / Actual
            # If Actual < Expected (Shortage), Ratio > 1.0 -> Price UP
            # If Actual > Expected (Surplus), Ratio < 1.0 -> Price DOWN
            
            # Safety: Avoid divide by zero
            current_biomass = np.maximum(biomass_paths[t, :], 1e-6)
            expected_biomass = market_consensus_biomass[t]
            
            supply_ratio = expected_biomass / current_biomass
            
            # Fundamental Value based on Scarcity
            fund_price = self.S0 * np.power(supply_ratio, scarcity_factor)
            
            # --- C. Add Market Noise (Random Walk) ---
            # Price isn't perfectly efficient; it wanders.
            drift_noise = (self.r - 0.5 * sigma_t**2) * dt
            diff_noise = sigma_t * Z1[t, :] * sqrt_dt
            
            # Accumulate the noise factor
            # noise_t = noise_{t-1} * exp(drift + diffusion)
            step_noise = np.exp(drift_noise + diff_noise)
            
            # For the first step, use base noise (1.0). For subsequent, multiply.
            # Actually, simpler: We apply the noise relative to the fundamental path.
            # But we need the noise to be cumulative (Random Walk).
            if t == 1:
                noise_accum[t, :] = noise_accum[t-1, :] * step_noise
            else:
                noise_accum[t, :] = noise_accum[t-1, :] * step_noise
            
            # Combine Fundamental + Noise
            S[t, :] = fund_price * noise_accum[t, :]
            
        return S, None # We don't return delta matrix anymore
