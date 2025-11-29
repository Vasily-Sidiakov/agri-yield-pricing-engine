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
        name_lower = name.lower()
        if 'soy' in name_lower:
            defaults['WP'] = 18.0  # C3 crops have lower water efficiency
            defaults['HI'] = 0.35
        elif 'wheat' in name_lower:
            defaults['WP'] = 15.0
            defaults['TAW'] = 140.0
        elif 'coffee' in name_lower:
            defaults['WP'] = 22.0
            defaults['CGC'] = 0.003 # Grows slower
            defaults['HI'] = 0.40   # Berry/Bean ratio
            
        return defaults

    def solve_odes(self, weather_paths, days_ahead):
        """
        Solves the system of differential equations over the simulation horizon.
        
        Inputs:
        - weather_paths: shape (days, n_paths) -> Simulated Tmax
        
        Returns:
        - yield_paths: shape (n_paths,) -> Final projected yield for each scenario
        - ks_history: shape (days, n_paths) -> Daily Ks stress factor
        """
        n_paths = weather_paths.shape[1]
        
        # 1. Initialize State Vectors
        B = np.zeros((days_ahead, n_paths))
        S = np.ones((days_ahead, n_paths)) * self.params['TAW'] * 0.8 # Start at 80% capacity
        CC = np.ones((days_ahead, n_paths)) * 0.01 # Start with seedlings/bud break
        
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
        for t in range(1, days_ahead):
            # Previous States
            S_prev = S[t-1, :]
            CC_prev = CC[t-1, :]
            B_prev = B[t-1, :]
            
            # --- A. Water Stress Coefficient (Ks) ---
            Dr = TAW - S_prev
            RAW = p_up * TAW 
            
            Ks = np.ones(n_paths)
            stressed_indices = Dr > RAW
            
            if np.any(stressed_indices):
                Ks[stressed_indices] = (TAW - Dr[stressed_indices]) / (TAW - RAW)
                Ks[stressed_indices] = np.maximum(Ks[stressed_indices], 0)
            
            Ks_history[t, :] = Ks
            
            # --- B. ODE 1: Canopy Cover (dCC/dt) ---
            if t < (days_ahead * 0.7):
                dCC = CGC * CC_prev * (CC_max - CC_prev) * Ks
            else:
                dCC = -CDC * CC_prev
            
            CC_new = np.clip(CC_prev + dCC, 0, CC_max)
            CC[t, :] = CC_new
            
            # --- C. ODE 2: Soil Moisture (dS/dt) ---
            # Stochastic Rain: Poisson process
            # Reduced probability to 10% to generate realistic drought scenarios
            rain_prob = 0.10 
            is_raining = np.random.rand(n_paths) < rain_prob
            rain_amount = np.random.exponential(15, size=n_paths) * is_raining
            
            current_tmax = weather_paths[t, :]
            ETo = 0.15 * current_tmax # Hargreaves Proxy
            
            Transpiration = ETo * (CC_new * self.params['Kcb']) * Ks
            
            dS = rain_amount - Transpiration
            S_new = np.clip(S_prev + dS, 0, TAW)
            S[t, :] = S_new
            
            # --- D. ODE 3: Biomass (dB/dt) ---
            dB = WP * CC_new * Ks
            
            # --- E. THERMAL SHOCK (Frost/Heat) ---
            # This introduces variance even if water is perfect
            
            # 1. Estimate Tmin (Diurnal range approx 12C)
            current_tmin = current_tmax - 12.0
            
            # 2. Coffee Frost Logic
            if 'coffee' in self.crop_name.lower():
                # If Tmin < 2C, tree takes damage
                frost_hit = current_tmin < 2.0
                if np.any(frost_hit):
                    # Biomass destruction (leaves/buds freeze)
                    # 5% loss per frost event
                    dB[frost_hit] -= (B_prev[frost_hit] * 0.05)
            
            # 3. Corn Heat Logic
            elif 'corn' in self.crop_name.lower():
                # If Tmax > 35C, pollination failure
                heat_hit = current_tmax > 35.0
                if np.any(heat_hit):
                    dB[heat_hit] *= 0.5 # Growth stunted by 50%
            
            B[t, :] = B_prev + dB
            
        # 3. Final Yield
        final_biomass = B[-1, :]
        final_yield = final_biomass * self.params['HI']
        
        return final_yield, Ks_history
