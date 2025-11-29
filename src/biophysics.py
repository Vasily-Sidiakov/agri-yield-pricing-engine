import numpy as np
import pandas as pd

class BiophysicalTwin:
    """
    Implements the Mechanistic Growth Model (AquaCrop-OS structure).
    Simulates crop development using coupled Ordinary Differential Equations (ODEs).
    
    UPDATED: Now driven by Vapor Pressure Deficit (VPD) to link atmospheric demand
    to soil moisture depletion, as per Section 2.2 of the research paper.
    """
    
    def __init__(self, commodity_name):
        self.crop_name = commodity_name.lower()
        self.params = self._get_crop_parameters(self.crop_name)
        
    def _get_crop_parameters(self, name):
        """
        Returns biological constants for specific crops.
        Source: FAO AquaCrop Reference Manual.
        """
        defaults = {
            'WP': 33.0,      # Water Productivity (g/m2)
            'TAW': 180.0,    # Total Available Water in root zone (mm)
            'p_up': 0.5,     # Upper threshold for moisture stress
            'CC_max': 0.95,  # Max canopy cover
            'CGC': 0.012,    # Canopy Growth Coefficient
            'CDC': 0.005,    # Canopy Decline Coefficient
            'HI': 0.48,      # Harvest Index 
            'Kcb': 1.05,     # Transpiration coeff
            't_base': 8.0,
            't_opt': 30.0,
            't_max': 45.0
        }
        
        if 'soy' in name:
            defaults.update({'WP': 18.0, 'HI': 0.35, 't_opt': 29.0, 't_max': 40.0})
        elif 'wheat' in name:
            defaults.update({'WP': 15.0, 'TAW': 140.0, 't_base': 0.0, 't_opt': 22.0, 't_max': 35.0})
        elif 'coffee' in name:
            defaults.update({'WP': 22.0, 'CGC': 0.003, 'HI': 0.40, 't_base': 15.0, 't_opt': 24.0, 't_max': 32.0})
        elif 'cocoa' in name:
            defaults.update({'WP': 20.0, 'CGC': 0.004, 't_base': 18.0, 't_opt': 25.0, 't_max': 33.0})
        elif 'cotton' in name:
            defaults.update({'WP': 16.0, 't_opt': 32.0, 't_max': 40.0, 'HI': 0.35})
        elif 'rice' in name:
            defaults.update({'WP': 19.0, 't_opt': 30.0, 't_base': 12.0, 'p_up': 0.2})
        elif 'sugar' in name:
            defaults.update({'WP': 35.0, 'HI': 0.70, 't_opt': 32.0, 't_max': 45.0})
            
        return defaults

    def solve_odes(self, tmax_paths, vpd_paths, days_ahead):
        """
        Solves the biophysical equations.
        NOW REQUIRES: vpd_paths (The "Thirst" of the atmosphere).
        """
        n_paths = tmax_paths.shape[1]
        
        # 1. Initialize State Vectors
        B = np.zeros((days_ahead, n_paths))
        S = np.ones((days_ahead, n_paths)) * self.params['TAW'] * 0.85 
        CC = np.ones((days_ahead, n_paths)) * 0.01 
        
        Ks_history = np.ones((days_ahead, n_paths))
        HI_damage = np.zeros(n_paths)
        
        vernalization_days = np.zeros(n_paths)
        
        # Parameters
        WP, TAW, p_up = self.params['WP'], self.params['TAW'], self.params['p_up']
        CC_max, CGC, CDC = self.params['CC_max'], self.params['CGC'], self.params['CDC']
        Kcb = self.params['Kcb']
        
        # 2. Time Integration Loop
        for t in range(1, days_ahead):
            S_prev = S[t-1, :]
            CC_prev = CC[t-1, :]
            B_prev = B[t-1, :]
            
            # --- A. Water Stress Coefficient (Ks) ---
            # Driven by Soil Moisture depletion (Dr)
            Dr = TAW - S_prev
            RAW = p_up * TAW 
            Ks = np.ones(n_paths)
            stressed_indices = Dr > RAW
            
            if np.any(stressed_indices):
                # Eq 2.3 from Document: Non-linear stress decay
                Ks[stressed_indices] = (TAW - Dr[stressed_indices]) / (TAW - RAW)
                Ks[stressed_indices] = np.maximum(Ks[stressed_indices], 0)
            
            if 'rice' in self.crop_name:
                dry_paddy = S_prev < (TAW * 0.9)
                Ks[dry_paddy] *= 0.5 
            
            Ks_history[t, :] = Ks
            
            # --- B. Canopy Growth ---
            if t < (days_ahead * 0.7):
                dCC = CGC * CC_prev * (CC_max - CC_prev) * Ks
            else:
                dCC = -CDC * CC_prev
            CC_new = np.clip(CC_prev + dCC, 0, CC_max)
            CC[t, :] = CC_new
            
            # --- C. Soil Moisture Dynamics (dS/dt) ---
            # Stochastic Rain (Poisson Process)
            rain_prob = 0.15 
            is_raining = np.random.rand(n_paths) < rain_prob
            rain_amount = np.random.exponential(12, size=n_paths) * is_raining
            
            # === THE FIX: LINKING VPD TO TRANSPIRATION ===
            # Old Math: ETo = 0.15 * Tmax (Too stable)
            # New Math: ETo is driven by VPD (Atmospheric Demand)
            # 1 kPa VPD approx corresponds to 3-4 mm/day demand in active season
            current_vpd = vpd_paths[t, :]
            ETo = 2.5 * current_vpd 
            
            # Actual Transpiration = Reference ET * Canopy * Ks
            Transpiration = ETo * (CC_new * Kcb) * Ks
            
            # Soil Balance: Rain fills it, Transpiration empties it
            dS = rain_amount - Transpiration
            S_new = np.clip(S_prev + dS, 0, TAW)
            S[t, :] = S_new
            
            # --- D. Biomass Accumulation (dB/dt) ---
            t_base, t_opt, t_max = self.params['t_base'], self.params['t_opt'], self.params['t_max']
            current_tmax = tmax_paths[t, :]
            current_tmean = current_tmax - 6.0 
            current_tmin = current_tmax - 12.0
            
            # Beta Function for Thermal Efficiency
            t_bio = np.clip(current_tmean, t_base + 0.1, t_max - 0.1)
            exponent = (t_max - t_opt) / (t_opt - t_base)
            term1 = ((t_max - t_bio) / (t_max - t_opt)) * ((t_bio - t_base) / (t_opt - t_base)) ** exponent
            temp_efficiency = np.nan_to_num(term1, nan=0.0)
            
            dB = WP * CC_new * Ks * temp_efficiency
            B[t, :] = B_prev + dB
            
            # --- E. Specific Stress Events ---
            if 'wheat' in self.crop_name:
                chill_days = current_tmean < 10.0
                vernalization_days[chill_days] += 1
            elif 'cotton' in self.crop_name:
                warm_nights = current_tmin > 24.0
                HI_damage[warm_nights] += 0.05
            elif 'coffee' in self.crop_name:
                frost_hit = current_tmin < 2.0
                HI_damage[frost_hit] += 0.10
            elif 'corn' in self.crop_name or 'soy' in self.crop_name:
                heat_hit = current_tmax > 35.0
                HI_damage[heat_hit] += 0.20
                drought_hit = Ks < 0.2
                HI_damage[drought_hit] += 0.05

        if 'wheat' in self.crop_name:
            failed_vernalization = vernalization_days < 30
            HI_damage[failed_vernalization] += 0.50
            
        if 'sugar' in self.crop_name:
            HI_damage = HI_damage * 0.5

        HI_damage = np.minimum(HI_damage, 1.0)
        realized_HI = self.params['HI'] * (1.0 - HI_damage)
        final_yield = B[-1, :] * realized_HI
        
        return final_yield, Ks_history
