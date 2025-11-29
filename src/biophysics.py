import numpy as np
import pandas as pd

class BiophysicalTwin:
    """
    Implements the Mechanistic Growth Model (AquaCrop-OS structure).
    Simulates crop development using coupled Ordinary Differential Equations (ODEs).
    
    State Variables:
    - B_t:  Accumulated Biomass (tons/ha)
    - S_t:  Soil Moisture (mm)
    - CC_t: Canopy Cover (0.0 to 1.0)
    """
    
    def __init__(self, commodity_name):
        self.crop_name = commodity_name
        self.params = self._get_crop_parameters(commodity_name)
        
    def _get_crop_parameters(self, name):
        """
        Returns biological constants for specific crops.
        Source: FAO AquaCrop Reference Manual.
        """
        # Default generic C4 crop (like Corn)
        defaults = {
            'WP': 33.0,      # Water Productivity (g/m2)
            'TAW': 180.0,    # Total Available Water in root zone (mm)
            'p_up': 0.5,     # Upper threshold for moisture stress (depletion fraction)
            'CC_max': 0.95,  # Max canopy cover
            'CGC': 0.012,    # Canopy Growth Coefficient
            'CDC': 0.005,    # Canopy Decline Coefficient (Senescence)
            'HI': 0.48,      # Harvest Index (Biomass -> Yield conversion)
            'Kcb': 1.05      # Transpiration coeff at full cover
        }
        
        # Adjustments based on crop type
        if 'soy' in name.lower():
            defaults['WP'] = 18.0  # C3 crops have lower water efficiency
            defaults['HI'] = 0.35
        elif 'wheat' in name.lower():
            defaults['WP'] = 15.0
            defaults['TAW'] = 140.0
        elif 'coffee' in name.lower():
            defaults['WP'] = 22.0
            defaults['CGC'] = 0.003 # Grows slower
            
        return defaults

    def solve_odes(self, weather_paths, days_ahead):
        """
        Solves the system of differential equations over the simulation horizon.
        
        Inputs:
        - weather_paths: shape (days, n_paths) -> Simulated Tmax/VPD
        
        Returns:
        - yield_paths: shape (n_paths,) -> Final projected yield for each scenario
        - stress_paths: shape (days, n_paths) -> Daily Ks stress factor
        """
        n_paths = weather_paths.shape[1]
        
        # 1. Initialize State Vectors
        # B = Biomass, S = Soil Moisture, CC = Canopy Cover
        B = np.zeros((days_ahead, n_paths))
        S = np.ones((days_ahead, n_paths)) * self.params['TAW'] # Start at field capacity
        CC = np.ones((days_ahead, n_paths)) * 0.01 # Start with tiny seedlings
        
        # Stress Tracker (Ks)
        Ks_history = np.ones((days_ahead, n_paths))
        
        # Parameters
        WP = self.params['WP']
        TAW = self.params['TAW']
        p_up = self.params['p_up']
        CC_max = self.params['CC_max']
        CGC = self.params['CGC']
        CDC = self.params['CDC']
        
        # 2. Time Integration Loop (Forward Euler)
        # We step through day by day
        for t in range(1, days_ahead):
            # Previous States
            S_prev = S[t-1, :]
            CC_prev = CC[t-1, :]
            B_prev = B[t-1, :]
            
            # --- A. Calculate Water Stress Coefficient (Ks) ---
            # Eq 2.3: Non-linear stress function
            # If Soil Moisture < Threshold, Ks drops exponentially
            
            # Depletion level (Dr)
            Dr = TAW - S_prev
            RAW = p_up * TAW # Readily Available Water
            
            # Ks Calculation
            Ks = np.ones(n_paths)
            stressed_indices = Dr > RAW
            
            # If depleted > RAW, calculate stress curve
            # Ks = (TAW - Dr) / (TAW - RAW)
            if np.any(stressed_indices):
                Ks[stressed_indices] = (TAW - Dr[stressed_indices]) / (TAW - RAW)
                Ks[stressed_indices] = np.maximum(Ks[stressed_indices], 0) # Clamp at 0
            
            Ks_history[t, :] = Ks
            
            # --- B. ODE 1: Canopy Cover (dCC/dt) ---
            # Logistic Growth adjusted by Stress (Ks)
            # If t < half_season: Grow. If t > half_season: Senesce (Die)
            if t < (days_ahead * 0.7):
                dCC = CGC * CC_prev * (CC_max - CC_prev) * Ks
            else:
                dCC = -CDC * CC_prev # Decay phase
            
            CC_new = np.clip(CC_prev + dCC, 0, CC_max)
            CC[t, :] = CC_new
            
            # --- C. ODE 2: Soil Moisture (dS/dt) ---
            # dS = Precip - Transpiration
            # We assume random rain events (Poisson-like) for the simulation
            # Simplified: Random rain * probability
            rain_prob = 0.2
            is_raining = np.random.rand(n_paths) < rain_prob
            rain_amount = np.random.exponential(10, size=n_paths) * is_raining
            
            # Transpiration = ETo * Kc * Ks
            # We approximate ETo (Evapotranspiration) using Temp/VPD from weather paths
            # Higher VPD = Higher Demand
            current_weather_t = weather_paths[t, :] # Tmax proxy
            ETo = 0.15 * current_weather_t # Simple Hargreaves proxy
            
            Transpiration = ETo * (CC_new * self.params['Kcb']) * Ks
            
            dS = rain_amount - Transpiration
            S_new = np.clip(S_prev + dS, 0, TAW) # Cannot exceed Field Capacity
            S[t, :] = S_new
            
            # --- D. ODE 3: Biomass Accumulation (dB/dt) ---
            # dB = WP * (Transpiration/ETo)
            # Since Transpiration = ETo * CC * Ks, the ETo cancels out nicely
            # dB = WP * CC * Ks
            dB = WP * CC_new * Ks
            
            B[t, :] = B_prev + dB
            
        # 3. Final Yield Calculation
        # Yield = Biomass * Harvest Index
        final_biomass = B[-1, :]
        final_yield = final_biomass * self.params['HI']
        
        return final_yield, Ks_history
