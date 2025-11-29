import numpy as np
import pandas as pd

class BiophysicalTwin:
    """
    Implements the Mechanistic Growth Model (AquaCrop-OS structure).
    Simulates crop development using coupled Ordinary Differential Equations (ODEs).
    
    Now supports crop-specific phenology:
    - Vernalization (Wheat)
    - Anaerobic conditions (Rice)
    - Boll Shedding (Cotton)
    - Frost Damage (Coffee)
    - Heat Sterilization (Corn/Soy)
    """
    
    def __init__(self, commodity_name):
        self.crop_name = commodity_name.lower()
        self.params = self._get_crop_parameters(self.crop_name)
        
    def _get_crop_parameters(self, name):
        """
        Returns biological constants for specific crops.
        Source: FAO AquaCrop Reference Manual.
        """
        # Default generic C4 crop (like Corn)
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
        
        # --- CROP SPECIFIC OVERRIDES ---
        if 'soy' in name:
            defaults.update({'WP': 18.0, 'HI': 0.35, 't_opt': 29.0, 't_max': 40.0})
        elif 'wheat' in name:
            # Winter Wheat needs cold
            defaults.update({'WP': 15.0, 'TAW': 140.0, 't_base': 0.0, 't_opt': 22.0, 't_max': 35.0})
        elif 'coffee' in name:
            defaults.update({'WP': 22.0, 'CGC': 0.003, 'HI': 0.40, 't_base': 15.0, 't_opt': 24.0, 't_max': 32.0})
        elif 'cocoa' in name:
            # Tropical tree, steady growth
            defaults.update({'WP': 20.0, 'CGC': 0.004, 't_base': 18.0, 't_opt': 25.0, 't_max': 33.0})
        elif 'cotton' in name:
            # Heat tolerant but sensitive night temps
            defaults.update({'WP': 16.0, 't_opt': 32.0, 't_max': 40.0, 'HI': 0.35})
        elif 'rice' in name:
            # Needs saturation
            defaults.update({'WP': 19.0, 't_opt': 30.0, 't_base': 12.0, 'p_up': 0.2}) # Low p_up means it stresses easily if dry
        elif 'sugar' in name:
            # C4 Grass, very hardy
            defaults.update({'WP': 35.0, 'HI': 0.70, 't_opt': 32.0, 't_max': 45.0})
            
        return defaults

    def solve_odes(self, weather_paths, days_ahead):
        n_paths = weather_paths.shape[1]
        
        # 1. Initialize State Vectors
        B = np.zeros((days_ahead, n_paths))
        S = np.ones((days_ahead, n_paths)) * self.params['TAW'] * 0.85 
        CC = np.ones((days_ahead, n_paths)) * 0.01 
        
        Ks_history = np.ones((days_ahead, n_paths))
        HI_damage = np.zeros(n_paths)
        
        # Trackers for specific crop logic
        vernalization_days = np.zeros(n_paths) # For Wheat
        
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
            
            # RICE SPECIAL LOGIC: Penalize if NOT saturated
            if 'rice' in self.crop_name:
                # If soil moisture is less than 90% capacity, Rice is stressed
                dry_paddy = S_prev < (TAW * 0.9)
                Ks[dry_paddy] *= 0.5 # Severe penalty for lack of standing water
            
            Ks_history[t, :] = Ks
            
            # --- B. Canopy Growth ---
            if t < (days_ahead * 0.7):
                dCC = CGC * CC_prev * (CC_max - CC_prev) * Ks
            else:
                dCC = -CDC * CC_prev
            CC_new = np.clip(CC_prev + dCC, 0, CC_max)
            CC[t, :] = CC_new
            
            # --- C. Soil Moisture ---
            rain_prob = 0.15 
            is_raining = np.random.rand(n_paths) < rain_prob
            rain_amount = np.random.exponential(12, size=n_paths) * is_raining
            
            current_tmax = weather_paths[t, :]
            current_tmean = current_tmax - 6.0 
            current_tmin = current_tmax - 12.0 # Diurnal Approx
            
            ETo = 0.15 * current_tmax 
            Transpiration = ETo * (CC_new * Kcb) * Ks
            
            dS = rain_amount - Transpiration
            S_new = np.clip(S_prev + dS, 0, TAW)
            S[t, :] = S_new
            
            # --- D. Biomass Accumulation ---
            t_base, t_opt, t_max = self.params['t_base'], self.params['t_opt'], self.params['t_max']
            
            # Beta Function (Thermal Efficiency)
            t_bio = np.clip(current_tmean, t_base + 0.1, t_max - 0.1)
            exponent = (t_max - t_opt) / (t_opt - t_base)
            term1 = ((t_max - t_bio) / (t_max - t_opt)) * ((t_bio - t_base) / (t_opt - t_base)) ** exponent
            temp_efficiency = np.nan_to_num(term1, nan=0.0)
            
            dB = WP * CC_new * Ks * temp_efficiency
            B[t, :] = B_prev + dB
            
            # --- E. CROP SPECIFIC STRESS RULES ---
            
            # 1. WHEAT: Vernalization Requirement
            if 'wheat' in self.crop_name:
                # Accumulate days where T < 10C
                chill_days = current_tmean < 10.0
                vernalization_days[chill_days] += 1
                
            # 2. COTTON: Night Temperature Stress (Boll Shedding)
            elif 'cotton' in self.crop_name:
                # If nights are too warm (>24C), bolls fall off
                warm_nights = current_tmin > 24.0
                HI_damage[warm_nights] += 0.05
                
            # 3. COFFEE: Frost Damage
            elif 'coffee' in self.crop_name:
                frost_hit = current_tmin < 2.0
                HI_damage[frost_hit] += 0.10
                
            # 4. CORN/SOY: Heat Sterilization
            elif 'corn' in self.crop_name.lower():
                # LOWERED THRESHOLD: 35C -> 32C (More frequent stress)
                heat_hit = current_tmax > 32.0 
                # INCREASED PENALTY: 20% -> 30% per day
                HI_damage[heat_hit] += 0.30 
                
                drought_hit = Ks < 0.2
                HI_damage[drought_hit] += 0.05

        # 3. Post-Season Adjustments
        
        # WHEAT CHECK: Did it get enough cold?
        if 'wheat' in self.crop_name:
            # If less than 30 days of chill, yield crashes (failed vernalization)
            failed_vernalization = vernalization_days < 30
            HI_damage[failed_vernalization] += 0.50 # 50% penalty
            
        # SUGAR CHECK: Cane is resilient
        if 'sugar' in self.crop_name:
            # Sugar tolerates stress better, reduce damage
            HI_damage = HI_damage * 0.5

        # Final Calculation
        HI_damage = np.minimum(HI_damage, 1.0)
        realized_HI = self.params['HI'] * (1.0 - HI_damage)
        
        final_biomass = B[-1, :]
        final_yield = final_biomass * realized_HI
        
        return final_yield, Ks_history
