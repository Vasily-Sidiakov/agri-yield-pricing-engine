import numpy as np
import pandas as pd

class StochasticPricingEngine:
    """
    Implements a Structural Scarcity Model coupled with Seasonal Heston Volatility.
    
    CALIBRATION UPDATE: High-Signal / Low-Noise tuning.
    Ensures Fundamental Supply shocks overpower random market noise.
    """
    
    def __init__(self, current_price, risk_free_rate=0.04):
        self.S0 = current_price
        self.r = risk_free_rate
        
        # Heston Parameters (Dampened Noise)
        self.kappa_v = 1.5   
        self.theta_v = 0.04  # Lower baseline volatility (20%)
        self.xi_v = 0.2      # Lower 'Vol of Vol' to reduce noise
        self.rho = -0.7      

    def simulate_pricing_paths(self, biomass_paths, aad_paths, days_ahead):
        """
        Generates Futures Price Paths based on Structural Scarcity.
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
        # The market expects the "Average" outcome.
        market_consensus_biomass = np.mean(biomass_paths, axis=1)
        
        # Scarcity Factor (Lambda) - AMPLIFIED
        # 15.0 means a 1% drop in supply = 15% increase in Fundamental Price.
        # This ensures the signal is stronger than the Heston noise.
        scarcity_factor = 15.0 
        
        # Correlated Brownian Motions
        Z1 = np.random.normal(0, 1, (days_ahead, n_paths))
        Z2 = np.random.normal(0, 1, (days_ahead, n_paths))
        
        dW_v = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * sqrt_dt
        
        # Noise accumulator
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
            
            sigma_t = np.sqrt(v_new) * stress_multiplier
            
            # --- B. Fundamental Price (The Signal) ---
            current_biomass = np.maximum(biomass_paths[t, :], 1e-6)
            expected_biomass = market_consensus_biomass[t]
            
            # Ratio > 1.0 (Shortage) -> Price UP
            supply_ratio = expected_biomass / current_biomass
            
            # Fundamental Value
            fund_price = self.S0 * np.power(supply_ratio, scarcity_factor)
            
            # --- C. Market Noise (The Random Walk) ---
            drift_noise = (self.r - 0.5 * sigma_t**2) * dt
            diff_noise = sigma_t * Z1[t, :] * sqrt_dt
            
            step_noise = np.exp(drift_noise + diff_noise)
            
            if t == 1:
                noise_accum[t, :] = step_noise
            else:
                noise_accum[t, :] = noise_accum[t-1, :] * step_noise
            
            # Combine: Signal * Noise
            S[t, :] = fund_price * noise_accum[t, :]
            
        return S, None
