import numpy as np
import pandas as pd


class StochasticPricingEngine:
    """Pricing engine that links simulated biomass (supply) and weather stress
    to futures price dynamics.

    Design goals
    ------------
    * Keep the structural idea from the original project:
      - Fundamental price responds to supply (biomass) shocks.
      - Volatility is higher during periods of weather stress (AAD paths).
    * Avoid the numerical explosions that occurred when biomass was near zero
      and the model raised a huge supply ratio to a large power.
    * Produce price paths that live in a realistic range so that the
      yield→price regression used later in the pipeline is meaningful.

    Interface
    ---------
    Parameters
    ----------
    current_price : float
        Current futures price (spot used as proxy).
    risk_free_rate : float, default 0.04
        Annualized risk-free rate used for a simple forward drift.

    Methods
    -------
    simulate_pricing_paths(biomass_paths, aad_paths, days_ahead)
        Returns (price_paths, convenience_yields).
        In this implementation convenience_yields is set to None but the
        interface is preserved for future extension.
    """  # noqa: E501

    def __init__(self, current_price: float, risk_free_rate: float = 0.04) -> None:
        self.S0 = float(current_price)
        self.r = float(risk_free_rate)

        # Heston-style variance dynamics (kept but heavily damped)
        self.kappa_v = 1.5   # Speed of mean reversion
        self.theta_v = 0.04  # Long-run variance (20% annual vol squared)
        self.xi_v = 0.2      # Vol of vol (kept modest)
        self.rho = -0.7      # Correlation between price and volatility shocks

        # Supply-elasticity parameter:
        # For a 10% supply shortfall near harvest, the structural component
        # will move prices on the order of ~15–25%, before adding noise.
        self.lambda_supply = 2.0

        # Cap variance and supply multipliers to prevent numerical blow-ups.
        self.min_var = 1e-5          # ~1% annual vol floor
        self.max_var = 0.50          # ≈70% annual vol ceiling
        self.min_fund_mult = 0.4     # Price cannot fall below 40% of base from fundamentals alone
        self.max_fund_mult = 2.5     # Or rise above 2.5× from fundamentals alone

    def _normalize_inputs(self, biomass_paths, aad_paths, days_ahead):
        """Validate and align input arrays.

        Parameters
        ----------
        biomass_paths : array-like, shape (T, n_paths)
            Proxy for cumulative biomass or supply for each Monte Carlo path.
        aad_paths : array-like, shape (T, n_paths)
            Accumulated atmospheric demand / stress measure.
        days_ahead : int
            Number of trading days to simulate.

        Returns
        -------
        biomass, aad, T, n_paths
        """  # noqa: D401
        biomass = np.asarray(biomass_paths, dtype=float)
        aad = np.asarray(aad_paths, dtype=float)

        if biomass.ndim != 2:
            raise ValueError("biomass_paths must be a 2D array (time × paths).")
        if aad.ndim != 2:
            raise ValueError("aad_paths must be a 2D array (time × paths).")

        T, n_paths = biomass.shape
        if aad.shape[0] != T or aad.shape[1] != n_paths:
            raise ValueError("biomass_paths and aad_paths must have the same shape.")

        if days_ahead is None:
            days_ahead = T
        else:
            days_ahead = int(days_ahead)
            if days_ahead <= 1:
                raise ValueError("days_ahead must be at least 2.")
            days_ahead = min(days_ahead, T)

        return biomass[:days_ahead, :], aad[:days_ahead, :], days_ahead, n_paths

    def simulate_pricing_paths(self, biomass_paths, aad_paths, days_ahead):
        """Simulate futures price paths.

        The price at each time t is decomposed into:
            S_t = F_t * F_supply(t) * N_t

        where:
            F_t       = risk-free forward from S0,
            F_supply  = structural multiplier from supply (biomass) shocks,
            N_t       = stochastic noise from a damped Heston process,
                        with volatility modestly amplified by daily stress.

        Returns
        -------
        prices : ndarray of shape (days_ahead, n_paths)
            Simulated futures prices.
        convenience_yields : None
            Placeholder to keep the original interface intact.
        """  # noqa: D401
        biomass, aad, days_ahead, n_paths = self._normalize_inputs(
            biomass_paths, aad_paths, days_ahead
        )

        dt = 1.0 / 252.0
        sqrt_dt = np.sqrt(dt)

        # Arrays for prices and variance
        S = np.zeros((days_ahead, n_paths), dtype=float)
        v = np.zeros((days_ahead, n_paths), dtype=float)

        # Initial conditions
        S[0, :] = self.S0
        v[0, :] = max(self.theta_v, self.min_var)

        # Simple risk-free forward curve as a baseline
        t_grid = np.arange(days_ahead, dtype=float)
        forward_base = self.S0 * np.exp(self.r * dt * t_grid)  # shape (days_ahead,)

        # Baseline (expected) biomass per day across paths.
        # We clip to avoid dividing by zero later.
        expected_biomass = np.mean(biomass, axis=1)
        expected_biomass = np.maximum(expected_biomass, 1e-3)

        # Pre-draw Brownian increments for the variance process
        Z1 = np.random.normal(0.0, 1.0, size=(days_ahead, n_paths))
        Z2 = np.random.normal(0.0, 1.0, size=(days_ahead, n_paths))
        dW_v = (self.rho * Z1 + np.sqrt(1.0 - self.rho**2) * Z2) * sqrt_dt

        # Log of the noise component; starts at 0 so noise_mult[0] = 1
        log_noise = np.zeros((days_ahead, n_paths), dtype=float)

        # --- Weather-stress link to volatility ---
        # Use DAILY increment in AAD so this doesn't explode with horizon length.
        aad_diff = np.zeros_like(aad)
        aad_diff[1:, :] = np.maximum(aad[1:, :] - aad[:-1, :], 0.0)

        # Normalize by a high percentile so that most days live in [0, 1]
        stress_scale = np.percentile(aad_diff, 95)
        if not np.isfinite(stress_scale) or stress_scale <= 0:
            stress_norm = np.zeros_like(aad_diff)
        else:
            stress_norm = np.clip(aad_diff / (2.0 * stress_scale), 0.0, 1.0)

        for t in range(1, days_ahead):
            # --- 1) Variance (Heston-style but heavily clipped) ---
            v_prev = np.maximum(v[t - 1, :], self.min_var)

            v_t = (
                v_prev
                + self.kappa_v * (self.theta_v - v_prev) * dt
                + self.xi_v * np.sqrt(v_prev) * dW_v[t, :]
            )
            v_t = np.clip(v_t, self.min_var, self.max_var)
            v[t, :] = v_t

            sigma_t = np.sqrt(v_t)

            # Volatility amplification during high-stress days
            vol_mult = 1.0 + 0.5 * stress_norm[t, :]  # up to +50% on extreme days
            sigma_eff = sigma_t * vol_mult

            # --- 2) Structural supply component ---
            current_biomass = np.maximum(biomass[t, :], 1e-3)
            base_b = expected_biomass[t]

            # Relative supply deviation (positive when supply is BELOW normal,
            # i.e., bullish for price). Clamp to ±30%.
            delta_supply = (base_b - current_biomass) / base_b
            delta_supply = np.clip(delta_supply, -0.30, 0.30)

            # Let the structural effect grow with time-to-harvest in a concave way.
            time_factor = np.sqrt(t / float(days_ahead))
            log_fund_mult = self.lambda_supply * delta_supply * time_factor

            fund_mult = np.exp(log_fund_mult)
            fund_mult = np.clip(fund_mult, self.min_fund_mult, self.max_fund_mult)

            # --- 3) Stochastic noise (GBM on the noise term only) ---
            drift_noise = (self.r - 0.5 * sigma_eff**2) * dt
            diff_noise = sigma_eff * Z1[t, :] * sqrt_dt
            log_noise[t, :] = log_noise[t - 1, :] + drift_noise + diff_noise

            noise_mult = np.exp(log_noise[t, :])

            # --- 4) Combine components ---
            S[t, :] = forward_base[t] * fund_mult * noise_mult

        # Convenience yields are not explicitly modeled in this implementation,
        # but the return signature keeps a placeholder for future extensions.
        convenience_yields = None

        return S, convenience_yields
