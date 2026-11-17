"""Shared, environment-independent reward-shaping formulas."""

import config


def cp_etr_reward(env_reward, phi_before, phi_after,
                  alpha=config.CP_ETR_SHAPING_COEFF,
                  gamma=config.DISCOUNT_FACTOR):
    """Equation (8): raw reward plus the discounted CP-ETR difference."""
    return float(env_reward) + alpha * (gamma * phi_after - phi_before)
