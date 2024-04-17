import numpy as np
from scipy.stats import norm


def calculate_psr(returns, rf=0, sr_star=0, periods_per_year=365):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    annualized_return = mean_return * periods_per_year
    annualized_std = std_dev * np.sqrt(periods_per_year)
    SR = (annualized_return - rf) / annualized_std
    T = len(returns)
    skewness = (np.mean((returns - mean_return) ** 3)) / std_dev**3
    kurtosis = (np.mean((returns - mean_return) ** 4)) / std_dev**4

    numerator = (SR - sr_star) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skewness * SR + ((kurtosis - 1) / 4) * SR**2)
    psr = norm.cdf(numerator / denominator)
    return psr


def calculate_sr(returns, rf=0, periods_per_year=365):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    annualized_return = mean_return * periods_per_year
    annualized_std = std_dev * np.sqrt(periods_per_year)
    SR = (annualized_return - rf) / annualized_std
    return SR


def calculate_avg_trade_hold_time_ms(trades):
    if len(trades) == 0:
        return 0

    cumulative_time_ms = 0

    for item in trades:
        cumulative_time_ms += item["close_time"] - item["open_time"]

    return cumulative_time_ms / len(trades)
