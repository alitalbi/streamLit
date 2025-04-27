import numpy as np
import pandas as pd

def compute_fractal_dimension(price_series: pd.Series, scaling_factor: int) -> pd.Series:
    log_returns = np.log(price_series / price_series.shift(1))
    R_i_n = np.log(price_series / price_series.shift(scaling_factor))
    N_i_n = log_returns.abs().rolling(scaling_factor).sum() / (R_i_n.abs() / scaling_factor)
    D_i_n = np.log(N_i_n) / np.log(scaling_factor)
    return D_i_n