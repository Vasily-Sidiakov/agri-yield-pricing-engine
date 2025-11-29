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
            'Kcb': 1.05,     # Transpiration coeff at full cover
            
            # THERMAL PARAMETERS
            't_base': 8.0,   # Lowered slightly for sensitivity
            't_opt': 30.0,
            't_max': 45.0
        }
        
        # Adjustments based on crop type
        name_lower = name.lower()
        if 'soy' in name_lower:
            defaults.update({'WP': 18.0, 'HI': 0.35, 't_opt': 29.0, 't_max': 40.0})
        elif 'wheat' in name_lower:
            defaults.update({'WP': 15.0, 'TAW': 140.0, 't_base': 0.0, 't_opt': 22.0, 't_max': 35.0})
        elif 'coffee' in name_lower:
            defaults.update({'WP': 22.0, 'CGC': 0.003, 'HI': 0.40, 't_base': 15.0, 't_opt': 24.0, 't_max': 32.0})
        elif 'cocoa' in name_lower:
            defaults.update({'WP': 20.0, 't_base': 18.0, 't_opt': 25.0, 't_max': 33.0})
        elif 'cotton' in name_lower:
            defaults.update({'WP': 16.0, 't_opt': 32.0, 't_max': 40.0})
            
        return defaults

    def calculate_beta_function(self, t_val, t_base, t_opt, t_max):
        """Calculates Thermal Efficiency (0.0 to 1.0)."""
        if t_val <= t_base or t_val >= t_max: return 0.0
        try:
            exponent = (t_max - t_opt) / (t_opt - t_base)
            term1 = ((t_max - t_val) / (t_max - t_opt)) * ((t_val - t_base) / (t_opt - t_base)) ** exponent
            return np.maximum(0.0, term1)
        except: return 0.0

    def solve_odes(self, weather_paths, days_ahead):
        """
        Solves the system of differential equations over the simulation horizon.
        Now includes PERMANENT YIELD DAMAGE logic.
        """
        n_paths = weather_paths.shape[1]
        
        # 1. Initialize State Vectors
        B = np.zeros((days_ahead, n_paths))
        S = np.ones((days_ahead, n_paths)) * self.params['TAW'] * 0.85 
        CC = np.ones((days_ahead, n_paths)) * 0.01 
        
        Ks_history = np.ones((days_ahead, n_paths))
        
        # === NEW: HARVEST INDEX PENALTY ACCUMULATOR ===
        # Starts at 0% damage. Accumulates if extreme events occur.
        HI_damage = np.zeros(n_paths)
        
        # Parameters
        WP, TAW, p_up = self.params['WP'], self.params['TAW'], self.params['p_up']
        CC_max, CGC, CDC = self.params['CC_max'], self.params['CGC'], self.params['CDC']
        Kcb = self.params['Kcb']
        
        # 2. Time Integration Loop
        for t in range(1, days_ahead):
            S_prev = S[t-1, :]
            CC_prev = CC[t-1, :]
            B_prev = B[t-1, :]
            
            # --- A. Water Stress (Ks) ---
            Dr = TAW - S_prev
            RAW = p_up * TAW 
            
            Ks = np.ones(n_paths)
            stressed_indices = Dr > RAW
            if np.any(stressed_indices):
                Ks[stressed_indices] = (TAW - Dr[stressed_indices]) / (TAW - RAW)
                Ks[stressed_indices] = np.maximum(Ks[stressed_indices], 0)
            
            Ks_history[t, :] = Ks
            
            # --- B. Canopy & Soil Dynamics ---
            if t < (days_ahead * 0.7):
                dCC = CGC * CC_prev * (CC_max - CC_prev) * Ks
            else:
                dCC = -CDC * CC_prev
            CC_new = np.clip(CC_prev + dCC, 0, CC_max)
            CC[t, :] = CC_new
            
            # Stochastic Rain
            rain_prob = 0.15 
            is_raining = np.random.rand(n_paths) < rain_prob
            rain_amount = np.random.exponential(12, size=n_paths) * is_raining
            
            current_tmax = weather_paths[t, :]
            current_tmean = current_tmax - 6.0 
            
            ETo = 0.15 * current_tmax 
            Transpiration = ETo * (CC_new * Kcb) * Ks
            S_new = np.clip(S_prev + rain_amount - Transpiration, 0, TAW)
            S[t, :] = S_new
            
            # --- C. Biomass Accumulation ---
            t_base, t_opt, t_max = self.params['t_base'], self.params['t_opt'], self.params['t_max']
            
            # Vectorized Beta Function
            t_bio = np.clip(current_tmean, t_base + 0.1, t_max - 0.1)
            exponent = (t_max - t_opt) / (t_opt - t_base)
            term1 = ((t_max - t_bio) / (t_max - t_opt)) * ((t_bio - t_base) / (t_opt - t_base)) ** exponent
            temp_efficiency = np.nan_to_num(term1, nan=0.0)
            
            dB = WP * CC_new * Ks * temp_efficiency
            B[t, :] = B_prev + dB
            
            # --- D. PERMANENT DAMAGE LOGIC (The Variance Booster) ---
            
            # 1. Coffee Frost (Tmin < 2C)
            if 'coffee' in self.crop_name.lower():
                current_tmin = current_tmax - 12.0
                frost_hit = current_tmin < 2.0
                # Permanent 10% Yield Loss per frost event
                HI_damage[frost_hit] += 0.10 
            
            # 2. Corn Heat Sterilization (Pollination Failure)
            elif 'corn' in self.crop_name.lower():
                # If Tmax > 35C, pollination fails regardless of water
                # Permanent 20% Yield Loss per day of extreme heat
                heat_hit = current_tmax > 35.0
                HI_damage[heat_hit] += 0.20 
                
                # Severe Drought Penalty (If Ks < 0.2)
                drought_hit = Ks < 0.2
                HI_damage[drought_hit] += 0.05

        # 3. Final Yield Calculation
        # Cap damage at 100% (Yield cannot be negative)
        HI_damage = np.minimum(HI_damage, 1.0)
        
        # Effective Harvest Index
        realized_HI = self.params['HI'] * (1.0 - HI_damage)
        
        final_biomass = B[-1, :]
        final_yield = final_biomass * realized_HI
        
        return final_yield, Ks_history
