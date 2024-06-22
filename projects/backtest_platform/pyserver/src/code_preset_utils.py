import json
from query_code_preset import CodePresetQuery


class CodePresetLabels:
    CANDLE_PATTERN = "candle_pattern"
    VOLATILITY = "volatility"
    PRICE_TRANSFORM = "price_transform"
    CYCLE = "cycle"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    SMOOTHING = "smoothing"
    OVERLAP = "overlap"
    CROSSING = "crossing"
    SEASONAL = "seasonal"
    TARGET = "target"
    CUSTOM = "custom"


class CodePresetCategories:
    INDICATOR = "backtest_create_columns"


class CodePreset:
    code: str
    name: str
    category: str
    description: str

    def __init__(self, code, name, category, description, labels: str) -> None:
        self.code = code
        self.name = name
        self.category = category
        self.description = description
        self.label = labels

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "label": self.label,
        }


GEN_RSI_CODE = """
def calculate_rsi(df, column='open_price', periods=[14]):
    for period in periods:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_label = f"RSI_{period}_{column}"
        df[df_label] = 100 - (100 / (1 + rs))

periods = [7, 14, 30, 70]
column = "MA_50_close_price"
calculate_rsi(dataset, column=column, periods=periods)
"""

GEN_MA_CODE = """
def calculate_ma(df, column='close_price', periods=[50]):
    for period in periods:
        ma_label = f"MA_{period}_{column}"
        df[ma_label] = df[column].rolling(window=period).mean()

periods = [20, 50, 100, 200]
column = "close_price"
calculate_ma(dataset, column=column, periods=periods)
"""

GEN_TARGETS = """
def create_targets(df, column='close_price', shifts=[7, 14, 30]):
    for shift in shifts:
        target_label = f"target_{shift}"
        df[target_label] = df[column].shift(-shift) / df[column]

shifts = [7, 14, 30]
create_targets(dataset, column='close_price', shifts=shifts)
"""


GEN_ATR = """
import pandas as pd
def calculate_atr(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[14]):
    for period in periods:
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - df[close_col].shift()).abs()
        low_close = (df[low_col] - df[close_col].shift()).abs()

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(window=period).mean()
        df_label = f"ATR_{period}"
        df[df_label] = atr


periods = [7, 14, 30, 70]
calculate_atr(dataset, periods=periods)
"""

IS_FRIDAY_8_UTC = """
import pandas as pd


dataset['kline_open_time_dt'] = pd.to_datetime(dataset['kline_open_time'], unit='ms')
dataset['is_friday_8_utc'] = dataset['kline_open_time_dt'].apply(
    lambda x: x.hour >= 8 and x.hour < 10 if x.weekday() == 4 else False
)
"""


GEN_OBV = """
def calculate_obv(df):
    df['OBV'] = 0
    for i in range(1, len(df)):
        if df['open_price'][i] > df['open_price'][i-1]:
            df['OBV'][i] = df['OBV'][i-1] + df['volume'][i]
        elif df['open_price'][i] < df['open_price'][i-1]:
            df['OBV'][i] = df['OBV'][i-1] - df['volume'][i]
        else:
            df['OBV'][i] = df['OBV'][i-1]

calculate_obv(dataset)
"""


GEN_HOURLY_FLAGS = """
import pandas as pd

def add_weekly_hourly_flags(df: pd.DataFrame, source_column: str) -> None:
    df[source_column + '_dt'] = pd.to_datetime(df[source_column], unit='ms')
    days_of_week = ['mon', 'tue', 'wed', 'thurs', 'fri', 'sat', 'sun']

    for day_idx, day_name in enumerate(days_of_week):
        for hour in range(24):
            column_name = f'is_{day_name}_{hour}_utc'  # Use the day name instead of the index
            df[column_name] = df[source_column + '_dt'].apply(
                lambda x: x.hour == hour if x.weekday() == day_idx else False
            )

add_weekly_hourly_flags(dataset, 'kline_open_time')
"""

GEN_SIMPLE_CROSSING = """
def calculate_crossing_indicator(df, column='value', threshold=90, direction='up'):
    if direction == 'up':
        crossed_label = f"crossed_{threshold}_up_{column}"
        df[crossed_label] = ((df[column] >= threshold) & (df[column].shift(1) < threshold)).astype(int)
    elif direction == 'down':
        crossed_label = f"crossed_{threshold}_down_{column}"
        df[crossed_label] = ((df[column] < threshold) & (df[column].shift(1) >= threshold)).astype(int)
    else:
        raise ValueError("Direction must be 'up' or 'down'")

calculate_crossing_indicator(dataset, "RSI_160_MA_200_close_price", 90, direction="up")
"""

GEN_PERSISTENT_CROSSING = """
def calculate_crossing_indicator_with_lookback(df, column='value', threshold=90, direction='up', look_back=5):
    crossed_label = f"crossed_{threshold}_{direction}_{column}"  # Define crossed_label before conditional branches
    if direction == 'up':
        initial_crosses = (df[column] >= threshold) & (df[column].shift(1) < threshold)
        for i in range(1, look_back + 1):
            initial_crosses &= df[column].shift(i) < threshold
    elif direction == 'down':
        initial_crosses = (df[column] < threshold) & (df[column].shift(1) >= threshold)
        for i in range(1, look_back + 1):
            initial_crosses &= df[column].shift(i) >= threshold
    else:
        raise ValueError("Direction must be 'up' or 'down'")

    df[crossed_label] = initial_crosses.astype(int)

calculate_crossing_indicator_with_lookback(dataset, "RSI_160_MA_200_close_price", 5, direction="down", look_back=160)
"""

GEN_BBANDS_CROSSING = """
import pandas as pd

def calculate_bbands_crossing(df, column='close_price', ma_period=20, std_factor=2, direction='up'):
    middle_band = df[column].rolling(window=ma_period).mean()
    
    std_dev = df[column].rolling(window=ma_period).std()
    
    upper_band = middle_band + (std_dev * std_factor)
    lower_band = middle_band - (std_dev * std_factor)
    
    if direction == 'up':
        crossed_label = f"bband_crossed_above_upper_{column}"
        df[crossed_label] = ((df[column] >= upper_band) & (df[column].shift(1) < upper_band)).astype(int)
    elif direction == 'down':
        crossed_label = f"bband_crossed_below_lower_{column}"
        df[crossed_label] = ((df[column] <= lower_band) & (df[column].shift(1) > lower_band)).astype(int)
    else:
        raise ValueError("Direction must be 'up' or 'down'")

calculate_bbands_crossing(df=dataset, column='close_price', ma_period=20, std_factor=2, direction='up')
"""

GEN_DEMA = """
import pandas as pd

def calculate_dema(df, column='close_price', periods=[20]):
    for period in periods:
        ema1 = df[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        dema_label = f"DEMA_{period}_{column}"
        df[dema_label] = dema

# Example usage:
periods = [20, 50, 100]
column = "close_price"
calculate_dema(dataset, column=column, periods=periods)
"""

GEN_EMA = """
import pandas as pd

def calculate_ema(df, column='close_price', periods=[12, 26]):
    for period in periods:
        ema_label = f"EMA_{period}_{column}"
        df[ema_label] = df[column].ewm(span=period, adjust=False).mean()

# Example usage
periods = [12, 26]
column = "close_price"
calculate_ema(dataset, column=column, periods=periods)
"""

HT_TRENDLINE = """
import numpy as np
import pandas as pd

def calculate_ht_trendline(df, column='close_price'):
    period = 40  # Commonly used period for HT_Trendline
    cycle_period = 0.075  # Commonly used in HT calculations
    
    # Detrend Price
    detrended = df[column] - df[column].rolling(window=period).mean()

    # Apply the Hilbert Transform to Detrended Data
    quadrature = detrended.shift(period // 2)
    inphase = detrended - quadrature

    # Compute the Instantaneous Phase
    # Prevent division by zero in case inphase equals zero
    inst_phase = np.where(inphase != 0, np.arctan(quadrature / inphase) / cycle_period, 0)
    df['HT_TRENDLINE'] = np.sin(inst_phase) * df[column]

calculate_ht_trendline(dataset, column='close_price')
"""


KAMA = """
import pandas as pd

def calculate_kama(df, column='close_price', period=10, fast=2, slow=30):
    change = df[column].diff(period).abs()
    volatility = df[column].diff().abs().rolling(window=period).sum()
    
    efficiency_ratio = change / volatility
    smoothing_constant = ((efficiency_ratio * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2).fillna(0)

    kama = pd.Series(index=df.index)
    kama.iloc[0] = df[column].iloc[0]
    
    for i in range(1, len(df)):
        kama.iloc[i] = kama.iloc[i - 1] + smoothing_constant.iloc[i] * (df[column].iloc[i] - kama.iloc[i - 1])

    df[f'KAMA_{period}'] = kama

# Configure the parameters
period = 10
fast = 2
slow = 30
column = 'close_price'
calculate_kama(dataset, column=column, period=period, fast=fast, slow=slow)
"""


MAVP = """
import pandas as pd

def calculate_mavp(df, column='close_price', period_col='variable_period'):
    if period_col not in df.columns or df[period_col].dtype not in ['int64', 'float64']:
        raise ValueError("period_col must exist in DataFrame and contain numeric values")

    mavp_label = f"mavp_variable_periods_{column}"
    df[mavp_label] = pd.Series(dtype='float64')

    for index, period in df[period_col].iteritems():
        if pd.notna(period) and period > 0:
            # Ensure the period is an integer and within a valid range
            period = int(period)
            # Calculate the mean for the period ending at the current index
            if index >= period - 1:
                df.at[index, mavp_label] = df.loc[index-period+1:index, column].mean()

    df.drop(columns=[period_col], inplace=True)

def default_period_callback(row, base_period):
    volatility_threshold = row['MA_50_close_price'] * 1.1
    return base_period + 2 if row['close_price'] > volatility_threshold else base_period

def generate_variable_periods(df, column='close_price', base_period=3, period_callback=None):
    if not callable(period_callback):
        raise ValueError("period_callback must be provided and should be a callable that takes a row and returns an integer")
    
    df['variable_period'] = df.apply(lambda row: period_callback(row, base_period), axis=1)



base_period = 24
generate_variable_periods(dataset, column='close_price', base_period=base_period, period_callback=default_period_callback)
calculate_mavp(dataset, column='close_price', period_col='variable_period')
"""

MAVP = """

import pandas as pd

def generate_variable_periods(df, base_period, callback):
    if not callable(callback):
        raise ValueError("callback must be a function")

    df['variable_period'] = df.apply(lambda row: callback(row, base_period), axis=1)

def calculate_mavp(df, price_col='close_price', period_col='variable_period'):
    mavp_label = f"mavp_{price_col}"
    df[mavp_label] = None  # Initialize column for moving averages

    for index in df.index:
        period = int(df.at[index, period_col])  
        if index >= period - 1:
            start_index = max(0, index - period + 1)  
            df.at[index, mavp_label] = df.loc[start_index:index, price_col].mean()  

    df.drop(columns=[period_col], inplace=True)  

def default_period_callback(row, base_period):
    volatility_threshold = row['MA_50_close_price'] * 1.1
    return base_period + 2 if row['close_price'] > volatility_threshold else base_period

# Example usage:
base_period = 24
generate_variable_periods(dataset, base_period=base_period, callback=default_period_callback)
calculate_mavp(dataset)
"""


MIDPOINT = """
import pandas as pd

def calculate_midpoint(df, column='close_price', periods=[14]):
    for period in periods:
        high = df[column].rolling(window=period).max()
        low = df[column].rolling(window=period).min()
        midpoint_label = f"MIDPOINT_{period}_{column}"
        df[midpoint_label] = (high + low) / 2

# Example usage:
periods = [14, 30, 60]
column = "close_price"
calculate_midpoint(dataset, column=column, periods=periods)
"""

MIDPRICE = """
def calculate_midprice(df, high_col='high_price', low_col='low_price', periods=[20]):
    for period in periods:
        midprice_label = f"MIDPRICE_{period}"
        df[midprice_label] = (df[high_col].rolling(window=period).max() + df[low_col].rolling(window=period).min()) / 2

# Example usage
periods = [20, 50, 100]
calculate_midprice(dataset, periods=periods)
"""


PARABOLIC_SAR = """
import pandas as pd

def calculate_sar(df, high_col='high_price', low_col='low_price', af_start=0.02, af_increment=0.02, af_max=0.2):
    sar = pd.Series(df[low_col])
    ep = df[high_col]  # extreme point
    trend = pd.Series(1, index=df.index)  # trend: 1 for uptrend, -1 for downtrend
    af = af_start

    for i in range(1, len(df)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = max(sar[i-1] + af * (ep[i-1] - sar[i-1]), df[low_col][i], df[low_col][i-1])
            if df[high_col][i] > ep[i-1]:
                ep[i] = df[high_col][i]
                af = min(af + af_increment, af_max)
            if df[low_col][i] < sar[i]:
                sar[i] = ep[i-1]
                trend[i] = -1
                af = af_start
                ep[i] = df[low_col][i]
        else:  # downtrend
            sar[i] = min(sar[i-1] + af * (ep[i-1] - sar[i-1]), df[high_col][i], df[high_col][i-1])
            if df[low_col][i] < ep[i-1]:
                ep[i] = df[low_col][i]
                af = min(af + af_increment, af_max)
            if df[high_col][i] > sar[i]:
                sar[i] = ep[i-1]
                trend[i] = 1
                af = af_start
                ep[i] = df[high_col][i]

    df['SAR'] = sar

# Usage example:
high_col = "high_price"
low_col = "low_price"
calculate_sar(dataset, high_col=high_col, low_col=low_col)
"""

SAREXT = """
import pandas as pd
def calculate_sarext(df, high_col='high_price', low_col='low_price', af_start=0.02, af_increment=0.02, af_max=0.2):
    length = len(df)
    sar = pd.Series(index=df.index)
    ep = pd.Series(index=df.index)
    af = pd.Series(index=df.index)
    trend = pd.Series(1, index=df.index)  # Start with an assumed uptrend

    # Initialize the first values
    sar[0] = df[low_col][0]
    ep[0] = df[high_col][0]
    af[0] = af_start

    for i in range(1, length):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if df[high_col][i] > ep[i-1]:
                ep[i] = df[high_col][i]
                af[i] = min(af[i-1] + af_increment, af_max)
            else:
                af[i] = af[i-1]
            
            if df[low_col][i] < sar[i]:
                sar[i] = ep[i-1]
                trend[i] = -1
                ep[i] = df[low_col][i]
                af[i] = af_start
            else:
                trend[i] = 1
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if df[low_col][i] < ep[i-1]:
                ep[i] = df[low_col][i]
                af[i] = min(af[i-1] + af_increment, af_max)
            else:
                af[i] = af[i-1]

            if df[high_col][i] > sar[i]:
                sar[i] = ep[i-1]
                trend[i] = 1
                ep[i] = df[high_col][i]
                af[i] = af_start
            else:
                trend[i] = -1

    df['SAREXT'] = sar

# Usage example:
high_col = "high_price"
low_col = "low_price"
calculate_sarext(dataset, high_col=high_col, low_col=low_col)
"""


T3_INDICATOR = """
import pandas as pd

def calculate_t3(df, column='close_price', periods=[10], volume_factor=0.7):
    for period in periods:
        e1 = df[column].ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        e4 = e3.ewm(span=period, adjust=False).mean()
        e5 = e4.ewm(span=period, adjust=False).mean()
        e6 = e5.ewm(span=period, adjust=False).mean()
        c1 = -volume_factor * volume_factor * volume_factor
        c2 = 3 * volume_factor * volume_factor + 3 * volume_factor * volume_factor * volume_factor
        c3 = -6 * volume_factor * volume_factor - 3 * volume_factor - 3 * volume_factor * volume_factor * volume_factor
        c4 = 1 + 3 * volume_factor + volume_factor * volume_factor * volume_factor + 3 * volume_factor * volume_factor
        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        t3_label = f"T3_{period}_{column}"
        df[t3_label] = t3

# Usage example:
periods = [10, 20, 30]
column = "close_price"
calculate_t3(dataset, column=column, periods=periods)
"""

TEMA_INDICATOR = """
import pandas as pd

def calculate_tema(df, column='close_price', periods=[30]):
    for period in periods:
        ema1 = df[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        tema_label = f"TEMA_{period}_{column}"
        df[tema_label] = 3 * (ema1 - ema2) + ema3

# Example usage:
column = "close_price"
periods = [20, 50, 100]
calculate_tema(dataset, column=column, periods=periods)
"""


TRIMA = """
import pandas as pd

def calculate_trima(df, column='close_price', period=30):
    n = period // 2 + 1
    trima_label = f"TRIMA_{period}_{column}"
    df[trima_label] = df[column].rolling(window=period).apply(lambda x: pd.Series(x).iloc[:n].sum() / n, raw=True)

column = "close_price"
period = 30
calculate_trima(dataset, column=column, period=period)
"""

WMA = """
import pandas as pd

def calculate_wma(df, column='close_price', periods=[10]):
    for period in periods:
        weights = pd.Series(range(1, period + 1))
        wma_label = f"WMA_{period}_{column}"
        df[wma_label] = df[column].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Usage example:
periods = [10, 20, 50]
column = "close_price"
calculate_wma(dataset, column=column, periods=periods)
"""

ADX = """
import pandas as pd

def calculate_adx(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=14):
    plus_dm = df[high_col].diff()
    minus_dm = df[low_col].diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - df[close_col].shift()).abs()
    tr3 = (df[low_col] - df[close_col].shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smoothed = tr.rolling(window=periods).sum()

    plus_di = 100 * (plus_dm.rolling(window=periods).sum() / tr_smoothed)
    minus_di = 100 * (minus_dm.rolling(window=periods).sum() / tr_smoothed)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=periods).mean()

    df['ADX'] = adx

# Usage example:
periods = 14
calculate_adx(dataset, periods=periods)
"""

ADXR = """
import pandas as pd

def calculate_adxr(df, period=14):
    # Calculate the True Range
    tr = pd.concat([df['high_price'] - df['low_price'],
                    (df['high_price'] - df['close_price'].shift()).abs(),
                    (df['low_price'] - df['close_price'].shift()).abs()], axis=1).max(axis=1)

    # Calculate the Plus and Minus Directional Movement
    plus_dm = df['high_price'].diff()
    minus_dm = df['low_price'].diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Smoothed values
    tr14 = tr.rolling(window=period, min_periods=1).sum()
    plus_dm14 = plus_dm.rolling(window=period, min_periods=1).sum()
    minus_dm14 = minus_dm.rolling(window=period, min_periods=1).sum()

    # Directional Indicators
    plus_di = 100 * (plus_dm14 / tr14)
    minus_di = 100 * (minus_dm14 / tr14)

    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()

    # ADXR
    adxr = (adx + adx.shift(period // 2)) / 2
    df['ADXR'] = adxr

# Usage example:
period = 14
calculate_adxr(dataset, period=period)
"""

APO = """
import pandas as pd

def calculate_apo(df, column='close_price', fast_period=12, slow_period=26):
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    df['APO_fast_{fast_period}_slow_{slow_period}_{column}'] = fast_ema - slow_ema

# Example usage:
column = "close_price"
fast_period = 12
slow_period = 26
calculate_apo(dataset, column=column, fast_period=fast_period, slow_period=slow_period)
"""


AROON_CUSTOM = """
import numpy as np
import pandas as pd

def calculate_aroon(df, column='close_price', periods=25):
    aroon_up = df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmax(x) + 1) / periods * 100, raw=True)
    aroon_down = df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmin(x) + 1) / periods * 100, raw=True)

    df[f'AROON_UP_CUST_{periods}_{column}'] = aroon_up
    df[f'AROON_DOWN_CUST_{periods}_{column}'] = aroon_down

# Usage example:
periods = 25
column = "close_price"
calculate_aroon(dataset, column=column, periods=periods)
"""

AROON_CONVENTIONAL = """
import numpy as np
import pandas as pd

def calculate_aroon(df, column='close_price', periods=[25]):
    for period in periods:
        # Calculate the rolling max and min for the given period
        rolling_max = df[column].rolling(window=period, min_periods=0).apply(lambda x: x.max())
        rolling_min = df[column].rolling(window=period, min_periods=0).apply(lambda x: x.min())

        # Identify the time since the last high and low within the window
        aroon_up = 100 * (period - df[column].rolling(window=period, min_periods=0).apply(
            lambda x: np.where(x[::-1] == x.max())[0][0] + 1)) / period
        aroon_down = 100 * (period - df[column].rolling(window=period, min_periods=0).apply(
            lambda x: np.where(x[::-1] == x.min())[0][0] + 1)) / period

        # Store the results in the dataframe
        df[f'AROON_UP_{period}_{column}'] = aroon_up
        df[f'AROON_DOWN_{period}_{column}'] = aroon_down

# Example usage:
periods = [14, 25, 50]
column = "close_price"
calculate_aroon(dataset, column=column, periods=periods)
"""

AROONOSC = """
import pandas as pd

def calculate_aroon_oscillator(df, high_col='high_price', low_col='low_price', period=25):
    aroon_up = 100 * df[high_col].rolling(window=period, min_periods=0).apply(lambda x: x[::-1].idxmax()) / (period - 1)
    aroon_down = 100 * df[low_col].rolling(window=period, min_periods=0).apply(lambda x: x[::-1].idxmin()) / (period - 1)
    aroon_osc = aroon_up - aroon_down
    df_label = f"AROON_OSCILLATOR_{period}"
    df[df_label] = aroon_osc

# Usage example:
period = 25
calculate_aroon_oscillator(dataset, period=period)
"""

BOP = """
import pandas as pd

def calculate_bop(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    bop = (df[close_col] - df[open_col]) / (df[high_col] - df[low_col])
    df['BOP'] = bop

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_bop(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CCI = """
import pandas as pd

def calculate_cci(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[20]):
    for period in periods:
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        sma = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())

        cci = (tp - sma) / (0.015 * mean_dev)
        cci_label = f"CCI_{period}"
        df[cci_label] = cci

# Usage example:
periods = [20, 40, 60]
calculate_cci(dataset, periods=periods)
"""

CMO = """
import pandas as pd

def calculate_cmo(df, column='close_price', period=14):
    delta = df[column].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()

    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    cmo_label = f"CMO_{period}_{column}"
    df[cmo_label] = cmo

# Example usage:
column = "close_price"
period = 14
calculate_cmo(dataset, column=column, period=period)
"""

DMI = """
import pandas as pd

def calculate_dmi(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=14):
    plus_dm = df[high_col].diff()
    minus_dm = df[low_col].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - df[close_col].shift()).abs()
    tr3 = (df[low_col] - df[close_col].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=periods).sum()
    plus_di = 100 * (plus_dm.rolling(window=periods).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=periods).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    
    adx = dx.rolling(window=periods).mean()

    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx

# Usage example:
periods = 14
calculate_dmi(dataset, periods=periods)
"""

MACD = """
import pandas as pd

def calculate_macd(df, column='close_price', short_period=12, long_period=26, signal_period=9):
    # Calculate the Short Term Exponential Moving Average
    short_ema = df[column].ewm(span=short_period, adjust=False).mean()

    # Calculate the Long Term Exponential Moving Average
    long_ema = df[column].ewm(span=long_period, adjust=False).mean()

    # Calculate the MACD line
    macd = short_ema - long_ema
    macd_label = f"MACD_{column}_short_{short_period}_long_{long_period}"
    df[macd_label] = macd

    # Calculate the signal line
    signal_label = f"MACD_Signal_{column}_short_{short_period}_long_{long_period}_signal_{signal_period}"
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    df[signal_label] = signal

    # Calculate MACD histogram
    histogram_label = f"MACD_Histogram_{column}_short_{short_period}_long_{long_period}_signal_{signal_period}"
    df[histogram_label] = df[macd_label] - df[signal_label]

# Example of how to use the function:
column = "close_price"
short_period = 12
long_period = 26
signal_period = 9
calculate_macd(dataset, column=column, short_period=short_period, long_period=long_period, signal_period=signal_period)
"""

MACDFIX = """
import pandas as pd

def calculate_macdfix(df, column='close_price', short_period=12, long_period=26, signal_period=9):
    ema_short = df[column].ewm(span=short_period, adjust=False).mean()
    ema_long = df[column].ewm(span=long_period, adjust=False).mean()
    
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    macd_label = f'MACD_{short_period}_{long_period}_{column}'
    signal_label = f'MACD_Signal_{signal_period}_{column}'
    
    df[macd_label] = macd
    df[signal_label] = macd_signal

# Example usage:
column = "close_price"
short_period = 12
long_period = 26
signal_period = 9
calculate_macdfix(dataset, column=column, short_period=short_period, long_period=long_period, signal_period=signal_period)
"""

MFI = """
import pandas as pd

def calculate_mfi(df, high_col='high_price', low_col='low_price', close_col='close_price', volume_col='volume', periods=[14]):
    for period in periods:
        # Typical Price
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Raw Money Flow
        raw_money_flow = typical_price * df[volume_col]
        
        # Money Flow Positive and Negative
        money_flow_positive = (typical_price > typical_price.shift(1)) * raw_money_flow
        money_flow_negative = (typical_price < typical_price.shift(1)) * raw_money_flow

        # Money Flow Ratio and MFI
        money_flow_ratio = money_flow_positive.rolling(window=period).sum() / money_flow_negative.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + money_flow_ratio))

        # Assign MFI to DataFrame
        df_label = f"MFI_{period}"
        df[df_label] = mfi

# Example usage:
periods = [14, 28, 42]  # Example periods for MFI calculation
calculate_mfi(dataset, periods=periods)
"""

MINUS_DI = """
def calculate_minus_di(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[14]):
    for period in periods:
        high = df[high_col]
        low = df[low_col]
        close_prev = df[close_col].shift(1)
        
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        tr_high = high - close_prev
        tr_low = close_prev - low
        
        true_range = pd.concat([high_diff, low_diff, tr_high, tr_low], axis=1).max(axis=1)
        
        high_low_diff = high.diff(periods=1)
        high_low_diff[high_low_diff < 0] = 0
        
        high_low_sum = high_low_diff.rolling(window=period).sum()
        
        tr_sum = true_range.rolling(window=period).sum()
        
        minus_di = (high_low_sum / tr_sum) * 100
        
        df_label = f"MINUS_DI_{period}"
        df[df_label] = minus_di

# Usage example:
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
periods = [14, 28, 50]
calculate_minus_di(dataset, high_col=high_col, low_col=low_col, close_col=close_col, periods=periods)
"""


MINUS_DM = """
import pandas as pd

def calculate_minus_dm(df, high_col='high_price', low_col='low_price', periods=[14]):
    for period in periods:
        previous_high = df[high_col].shift()
        previous_low = df[low_col].shift()

        minus_dm = (previous_high - df[high_col]).where((previous_high - df[high_col]) > (df[low_col] - previous_low), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        minus_dm_label = f"Minus_DM_{period}"
        df[minus_dm_label] = minus_dm.rolling(window=period).sum()

high_col = "high_price"
low_col = "low_price"
periods = [14, 28]
calculate_minus_dm(dataset, high_col=high_col, low_col=low_col, periods=periods)
"""

MOM = """
import pandas as pd

def calculate_momentum(df, column='close_price', periods=[10]):
    for period in periods:
        mom_label = f"MOM_{period}_{column}"
        df[mom_label] = df[column].diff(period)

# Usage example:
column = "close_price"
periods = [10, 15, 30]
calculate_momentum(dataset, column=column, periods=periods)
"""

PLUS_DI = """
import pandas as pd

def calculate_plus_di(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[14]):
    for period in periods:
        # Calculate the Plus Directional Movement (+DM)
        delta_high = df[high_col].diff()
        delta_low = df[low_col].diff()

        plus_dm = (delta_high > delta_low) & (delta_high > 0) * delta_high
        plus_dm.fillna(0, inplace=True)

        # Calculate the True Range (TR)
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - df[close_col].shift()).abs()
        low_close = (df[low_col] - df[close_col].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr_smooth = tr.rolling(window=period).sum()

        # Calculate the smoothed Plus Directional Indicator (+DI)
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        
        # Store the result in the dataframe
        df[f'PLUS_DI_{period}'] = plus_di

periods = [14, 28]
calculate_plus_di(dataset, periods=periods)
"""

PLUS_DM = """
import pandas as pd

def calculate_plus_dm(df, high_col='high_price', low_col='low_price', periods=[14]):
    for period in periods:
        plus_dm = df[high_col].diff()
        minus_dm = df[low_col].diff()

        # Calculate Plus Directional Movement
        mask = (plus_dm > minus_dm) & (plus_dm > 0)
        plus_dm = plus_dm.where(mask, 0.0)
        plus_dm_label = f"PLUS_DM_{period}"
        df[plus_dm_label] = plus_dm.rolling(window=period).sum()

# Usage example:
high_col = "high_price"
low_col = "low_price"
periods = [14]
calculate_plus_dm(dataset, high_col=high_col, low_col=low_col, periods=periods)
"""

PPO = """
import pandas as pd

def calculate_ppo(df, column='close_price', short_periods=[12], long_periods=[26], signal_period=9):
    for short_period in short_periods:
        for long_period in long_periods:
            ema_short = df[column].ewm(span=short_period, adjust=False).mean()
            ema_long = df[column].ewm(span=long_period, adjust=False).mean()
            
            ppo = 100 * ((ema_short - ema_long) / ema_long)
            ppo_label = f"PPO_{short_period}_{long_period}_{column}"
            df[ppo_label] = ppo

            signal_label = f"PPO_signal_{short_period}_{long_period}_{column}"
            df[signal_label] = ppo.ewm(span=signal_period, adjust=False).mean()

# Usage example:
column = "close_price"
short_periods = [12]
long_periods = [26]
signal_period = 9
calculate_ppo(dataset, column=column, short_periods=short_periods, long_periods=long_periods, signal_period=signal_period)
"""

ROC = """
import pandas as pd

def calculate_roc(df, column='close_price', periods=14):
    price = df[column]
    prev_price = df[column].shift(periods)
    roc = ((price / prev_price) - 1) * 100
    roc_label = f"ROC_{periods}_{column}"
    df[roc_label] = roc

# Usage example:
periods = 14
column = "close_price"
calculate_roc(dataset, column=column, periods=periods)
"""

ROCP = """
import pandas as pd

def calculate_rocp(df, column='close_price', periods=[1]):
    # First, ensure the column exists in DataFrame to avoid KeyError
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

    # Fill NaN values and ensure no infinite values exist in the column
    df[column] = df[column].replace([float('inf'), -float('inf')], pd.NA)
    df[column] = df[column].fillna(method='bfill').fillna(method='ffill')
    
    if df[column].isnull().any():
        raise ValueError("NaN values persist in the column after filling; check data integrity.")

    # Calculate ROCP for each specified period
    for period in periods:
        rocp_label = f"ROCP_{period}_{column}"
        # Shift and calculate difference
        shifted_data = df[column].shift(period)
        differences = df[column] - shifted_data
        # Safeguard against division by zero and inf values
        differences.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        shifted_data.replace(0, pd.NA, inplace=True)  # Replace zero to avoid division by zero

        # Calculate ROCP and handle potential NaN from division
        df[rocp_label] = (differences / shifted_data).fillna(0)

# Usage example:
periods = [1, 5, 10]  # You can specify multiple periods
column = "close_price"
calculate_rocp(dataset, column=column, periods=periods)
"""

ROCR = """
import pandas as pd

def calculate_rocr(df, column='close_price', periods=[10]):
    for period in periods:
        rocr_label = f"ROCR_{period}_{column}"
        df[rocr_label] = df[column] / df[column].shift(period)

# Usage example:
column = "close_price"
periods = [10, 20, 30]
calculate_rocr(dataset, column=column, periods=periods)
"""

ROCR100 = """
import pandas as pd

def calculate_rocr100(df, column='close_price', period=1):
    df['ROCR100'] = ((df[column] / df[column].shift(period)) * 100)

# Usage example:
column = "close_price"
period = 1
calculate_rocr100(dataset, column=column, period=period)
"""

STOCH = """
import pandas as pd

def calculate_stochastic(df, high_col='high_price', low_col='low_price', close_col='close_price', k_period=14, d_period=3):
    low_min = df[low_col].rolling(window=k_period).min()
    high_max = df[high_col].rolling(window=k_period).max()
    
    # Calculate %K line
    k_label = f'STOCHASTIC_K_{k_period}'
    df[k_label] = ((df[close_col] - low_min) / (high_max - low_min)) * 100
    
    # Calculate %D line as the moving average of %K
    d_label = f'STOCHASTIC_D_{k_period}_{d_period}'
    df[d_label] = df[k_label].rolling(window=d_period).mean()

# Configuration
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
k_period = 14
d_period = 3
calculate_stochastic(dataset, high_col=high_col, low_col=low_col, close_col=close_col, k_period=k_period, d_period=d_period)
"""

STOCHF = """
import pandas as pd

def calculate_stochf(df, high_col='high_price', low_col='low_price', close_col='close_price', k_period=14, k_smooth=3):
    # Calculate the raw Stochastic value (%K)
    lowest_low = df[low_col].rolling(window=k_period).min()
    highest_high = df[high_col].rolling(window=k_period).max()
    k_raw_label = f'StochF_raw_%K_{k_period}'
    df[k_raw_label] = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))

    # Smooth the %K value if required
    k_smooth_label = f'STOCHF_K_{k_period}_{k_smooth}'
    df[k_smooth_label] = df[k_raw_label].rolling(window=k_smooth).mean()

    # Calculate %D as moving average of %K
    d_label = f'STOCHF_D_{k_period}_{k_smooth}'
    df[d_label] = df[k_smooth_label].rolling(window=k_smooth).mean()

# Configuration
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
k_period = 14
k_smooth = 3
calculate_stochf(dataset, high_col=high_col, low_col=low_col, close_col=close_col, k_period=k_period, k_smooth=k_smooth)
"""

STOCHRSI = """
import pandas as pd

def calculate_stochrsi(df, column='close_price', periods=14, k_period=3, d_period=3):
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(window=k_period).min()
    max_rsi = rsi.rolling(window=k_period).max()
    
    stochrsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    
    stochrsi_k = stochrsi.rolling(window=k_period).mean()
    stochrsi_d = stochrsi_k.rolling(window=d_period).mean()

    # Including period and column name in the labels
    df[f'STOCHRSI_K_{periods}_{k_period}_{column}'] = stochrsi_k
    df[f'STOCHRSI_D_{periods}_{d_period}_{column}'] = stochrsi_d

# Example of usage:
column = "close_price"
periods = 14
k_period = 3
d_period = 3
calculate_stochrsi(dataset, column=column, periods=periods, k_period=k_period, d_period=d_period)
"""

TRIX = """
import pandas as pd

def calculate_trix(df, column='close_price', period=15):
    # First EMA
    ema1 = df[column].ewm(span=period, adjust=False).mean()
    # Second EMA
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    # Third EMA
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # TRIX is the 1-day percent change in the third EMA
    trix = ema3.pct_change() * 100
    trix_label = f"TRIX_{period}_{column}"
    df[trix_label] = trix

# Usage example
period = 15
column = "close_price"
calculate_trix(dataset, column=column, period=period)
"""

ULTOSC = """
import pandas as pd

def calculate_ultosc(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[7, 14, 28]):
    bp = df[close_col] - df[[low_col, close_col.shift()]].min(axis=1)
    tr = df[high_col] - df[low_col]
    tr = pd.concat([tr, (df[high_col] - df[close_col].shift()).abs(), (df[low_col] - df[close_col].shift()).abs()], axis=1).max(axis=1)
    
    avg_bp = {p: bp.rolling(window=p).sum() for p in periods}
    avg_tr = {p: tr.rolling(window=p).sum() for p in periods}
    
    ultosc = (4 * avg_bp[7] / avg_tr[7] + 2 * avg_bp[14] / avg_tr[14] + avg_bp[28] / avg_tr[28]) / (4 + 2 + 1) * 100
    df['ULTOSC'] = ultosc

# Define the columns and periods for your dataset
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
periods = [7, 14, 28]
calculate_ultosc(dataset, high_col=high_col, low_col=low_col, close_col=close_col, periods=periods)
"""

WILLR = """
import pandas as pd

def calculate_williams_r(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[14]):
    for period in periods:
        high = df[high_col].rolling(window=period).max()
        low = df[low_col].rolling(window=period).min()
        williams_r_label = f"WILLR_{period}"
        df[williams_r_label] = -100 * ((high - df[close_col]) / (high - low))

# Usage example:
periods = [14, 28, 56]  # Different periods for flexibility
calculate_williams_r(dataset, periods=periods)
"""

AD_LINE = """
import pandas as pd

def calculate_ad_line(df, close_col='close_price', high_col='high_price', low_col='low_price', volume_col='volume'):
    mfm = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
    mfv = mfm * df[volume_col]
    df['AD_Line'] = mfv.cumsum()

# Usage example:
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
volume_col = 'volume'
calculate_ad_line(dataset, close_col=close_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
"""

ADOSC = """
import pandas as pd

def calculate_adosc(df, high_col='high_price', low_col='low_price', close_col='close_price', volume_col='volume', short_period=3, long_period=10):
    # Calculate the Money Flow Multiplier
    mfm = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
    mfm.fillna(0, inplace=True)  # Handling division by zero

    # Calculate the Money Flow Volume
    mfv = mfm * df[volume_col]

    # Calculate the Chaikin A/D Line as a cumulative sum of Money Flow Volume
    ad = mfv.cumsum()

    # Calculate the short and long period Exponential Moving Averages of the A/D Line
    ad_ema_short = ad.ewm(span=short_period, adjust=False).mean()
    ad_ema_long = ad.ewm(span=long_period, adjust=False).mean()

    # The Chaikin Oscillator is the difference between the two EMAs
    oscillator_label = f'ADOSC_{short_period}_{long_period}'
    df[oscillator_label] = ad_ema_short - ad_ema_long

# Usage example
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
volume_col = 'volume'
short_period = 3
long_period = 10
calculate_adosc(dataset, high_col=high_col, low_col=low_col, close_col=close_col, volume_col=volume_col, short_period=short_period, long_period=long_period)
"""

HT_DCPERIOD = """
import pandas as pd
import numpy as np

def calculate_ht_dcperiod(df, price_col='close_price'):
    smooth_price = (4 * df[price_col] + 3 * df[price_col].shift(1) + 2 * df[price_col].shift(2) + df[price_col].shift(3)) / 10
    
    dpo = smooth_price - smooth_price.shift(int(0.5 * 20 + 1))  # Approximation for a cycle of period 20

    in_phase = 0.75 * (dpo.shift(2) - dpo.shift(3)) + 0.25 * (dpo.shift(1) - dpo.shift(4))
    quadrature = 0.5 * (dpo.shift(1) + dpo.shift(3)) - dpo

    ip = np.where(quadrature != 0, np.arctan(in_phase / quadrature) / (2 * np.pi), 0)
    dcperiod = 1 / np.abs(ip)

    df['HT_DCPERIOD'] = dcperiod

# Usage example:
price_col = "close_price"
calculate_ht_dcperiod(dataset, price_col=price_col)
"""

HT_DCPHASE = """
import numpy as np
import pandas as pd

def calculate_ht_dcphase(df, column='close_price'):
    # Calculate the Hilbert Transform - Dominant Cycle Phase
    period = 32
    cycle_period = np.zeros(len(df))
    inst_period = np.zeros(len(df))
    dc_phase = np.zeros(len(df))
    count = 0

    for i in range(period, len(df)):
        # Hilbert Transform
        real_part = np.sin(2 * np.pi / period)
        imag_part = np.cos(2 * np.pi / period)
        Q1 = df[column].iloc[i - period] * imag_part
        I1 = df[column].iloc[i - period] * real_part

        # Compute InPhase and Quadrature components
        jI = np.zeros((period,))
        jQ = np.zeros((period,))

        for j in range(1, period):
            jI[j] = jI[j - 1] + real_part * df[column].iloc[i - j] - I1
            jQ[j] = jQ[j - 1] + imag_part * df[column].iloc[i - j] - Q1

        Q2 = jQ[period - 1]
        I2 = jI[period - 1]

        # Dominant cycle phase
        dc_phase[i] = np.arctan2(Q2, I2) * (180 / np.pi)

    df_label = f'HT_DCPHASE_{column}'
    df[df_label] = dc_phase

# Usage example:
column = "close_price"
calculate_ht_dcphase(dataset, column=column)
"""

HT_PHASOR = """
import pandas as pd
import numpy as np
from scipy.signal import hilbert

def calculate_ht_phasor(df, column='close_price'):
    signal = df[column]
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_amplitude = np.abs(analytic_signal)
    
    df[f'HT_Phasor_Phase_{column}'] = instantaneous_phase
    df[f'HT_Phasor_Amplitude_{column}'] = instantaneous_amplitude

# Usage example:
column = "close_price"
calculate_ht_phasor(dataset, column=column)
"""

HT_SINE = """
import pandas as pd
import ta

def calculate_ht_sine(df, column='close_price'):
    sine, leadsine = ta.trend.ht_sine(df[column])
    df[f'HT_SINE_{column}'] = sine
    df[f'HT_LEADSINE_{column}'] = leadsine

# Usage example:
column = "close_price"
calculate_ht_sine(dataset, column=column)
"""

HT_TRENDMODE = """
import numpy as np
import pandas as pd

def calculate_ht_trendmode(df, column='close_price'):
    from scipy.signal import hilbert
    
    # Applying the Hilbert Transform to the price data to get the analytic signal
    analytic_signal = hilbert(df[column])
    # Calculating the instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # Determining the sine of the difference in phase
    sin_instantaneous_phase = np.sin(instantaneous_phase[1:] - instantaneous_phase[:-1])
    
    # Trend mode is where the sine of the phase difference is positive
    trend_mode = (sin_instantaneous_phase > 0).astype(int)
    # Append the initial value as np.nan because the difference calculation reduces the array size by 1
    trend_mode = np.insert(trend_mode, 0, np.nan)
    
    df[f'HT_TRENDMODE_{column}'] = trend_mode

# Example usage:
column = 'close_price'
calculate_ht_trendmode(dataset, column=column)
"""

AVGPRICE = """
import pandas as pd

def calculate_avgprice(df, high_col='high_price', low_col='low_price', close_col='close_price'):
    df['AVGPRICE'] = (df[high_col] + df[low_col] + df[close_col]) / 3

# Usage example:
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_avgprice(dataset, high_col=high_col, low_col=low_col, close_col=close_col)
"""

MEDPRICE = """
import pandas as pd

def calculate_medprice(df, high_col='high_price', low_col='low_price'):
    df['MEDPRICE'] = (df[high_col] + df[low_col]) / 2

# Usage example:
high_col = "high_price"
low_col = "low_price"
calculate_medprice(dataset, high_col=high_col, low_col=low_col)
"""

TYPPRICE = """
import pandas as pd

def calculate_typ_price(df, high_col='high_price', low_col='low_price', close_col='close_price'):
    df['TYPPRICE'] = (df[high_col] + df[low_col] + df[close_col]) / 3

# Usage example:
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_typ_price(dataset, high_col=high_col, low_col=low_col, close_col=close_col)
"""

WCLPRICE = """
import pandas as pd

def calculate_wclprice(df, high_col='high_price', low_col='low_price', close_col='close_price'):
    df['WCLPRICE'] = (df[high_col] + df[low_col] + 2 * df[close_col]) / 4

# Usage example:
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_wclprice(dataset, high_col=high_col, low_col=low_col, close_col=close_col)
"""

NATR = """
import pandas as pd

def calculate_natr(df, high_col='high_price', low_col='low_price', close_col='close_price', periods=[14]):
    for period in periods:
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - df[close_col].shift()).abs()
        low_close = (df[low_col] - df[close_col].shift()).abs()

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()

        natr = (atr / df[close_col]) * 100  # Normalization
        df_label = f"NATR_{period}"
        df[df_label] = natr

# Usage example:
periods = [14, 20, 50]
calculate_natr(dataset, periods=periods)
"""

TRANGE = """
import pandas as pd

def calculate_trange(df, high_col='high_price', low_col='low_price', close_col='close_price'):
    high_low = df[high_col] - df[low_col]
    high_close = (df[high_col] - df[close_col].shift()).abs()
    low_close = (df[low_col] - df[close_col].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['TRange'] = ranges.max(axis=1)

# Usage example:
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_trange(dataset, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL2CROWS = """
import pandas as pd

def calculate_cdl2crows(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Check for the conditions specific to the Two Crows pattern
    cdl2crows = []
    for i in range(2, len(df)):
        # First candle: bullish
        first_bullish = df[close_col][i-2] > df[open_col][i-2]
        # Second candle: gap up opening, closing within the range of the first candle but below its close
        second_gap_up = (df[open_col][i-1] > df[close_col][i-2]) and (df[close_col][i-1] < df[close_col][i-2]) and (df[close_col][i-1] > df[open_col][i-1])
        # Third candle: gap up opening, closing below the second candle's open
        third_bearish = (df[open_col][i] > df[open_col][i-1]) and (df[close_col][i] < df[open_col][i-1])

        if first_bullish and second_gap_up and third_bearish:
            cdl2crows.append(1)
        else:
            cdl2crows.append(0)
    
    df['CDL2CROWS'] = cdl2crows

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
calculate_cdl2crows(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLENGULFING = """
import pandas as pd

def calculate_engulfing_pattern(df, open_col='open_price', close_col='close_price'):
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)

    bullish_engulfing = ((df[close_col] > df['prev_open']) & 
                         (df[open_col] < df['prev_close']) & 
                         (df[close_col] > df['prev_close']) & 
                         (df[open_col] < df['prev_open']))

    bearish_engulfing = ((df[open_col] > df['prev_close']) & 
                         (df[close_col] < df['prev_open']) & 
                         (df[open_col] > df['prev_open']) & 
                         (df[close_col] < df['prev_close']))

    df['bullish_engulfing'] = bullish_engulfing.astype(int)
    df['bearish_engulfing'] = bearish_engulfing.astype(int)

# Usage example:
open_col = "open_price"
close_col = "close_price"
calculate_engulfing_pattern(dataset, open_col=open_col, close_col=close_col)
"""

CLIP_FROM_RANGE = """
import pandas as pd

def adjust_column_values_based_on_range(df, column_name, lower_bound, upper_bound):
    # Applying the conditional logic to the specified column
    df[column_name] = df[column_name].apply(lambda x: 0 if lower_bound <= x <= upper_bound else x)

adjust_column_values_based_on_range(dataset, 'ROCP_7_close_price', -30, 30)
"""

THRESHOLD_FLAG = """
import pandas as pd
import numpy as np

def set_threshold_flag(df, source_column, target_column, threshold, above_threshold=True):
    # Check if the source column exists to avoid KeyError
    if source_column not in df.columns:
        raise ValueError(f"The specified source column '{source_column}' does not exist in the DataFrame.")
    
    # Use numpy.where to set values in the new column based on the condition
    if above_threshold:
        df[target_column] = np.where(df[source_column] > threshold, 1, 0)
    else:
        df[target_column] = np.where(df[source_column] < threshold, 1, 0)

# Usage example:
source_column = "ROCP_168_volume"
target_column = "ROCP_168_volume_large_vol"
threshold = 1.0  # 100%
set_threshold_flag(dataset, source_column, target_column, threshold, above_threshold=True)
"""

DIFF_TWO = """
import pandas as pd

def calculate_diff(df, col1='column1', col2='column2'):
    diff_label = f"{col1}_minus_{col2}"
    df[diff_label] = df[col1] - df[col2]

col1 = "open_price"
col2 = "close_price"
calculate_diff(dataset, col1=col1, col2=col2)
"""

ROCP_DIFF = """
import pandas as pd

def calculate_percentage_change(df, col1='column1', col2='column2'):
    change_label = f"{col1}_perc_change_frm_{col2}"
    df[change_label] = ((df[col1] - df[col2]) / df[col2]) * 100


col1 = "open_price"
col2 = "close_price"
calculate_percentage_change(dataset, col1=col1, col2=col2)
"""

RATIO_DIFF = """
import pandas as pd

def calculate_ratio(df, col1='column1', col2='column2'):
    ratio_label = f"{col1}_to_{col2}_ratio"
    df[ratio_label] = df[col1] / df[col2]

# Usage example:
col1 = "open_price"
col2 = "close_price"
calculate_ratio(dataset, col1=col1, col2=col2)
"""

SUM_FUNC = """
import pandas as pd

def calculate_sum(df, col1='column1', col2='column2'):
    sum_label = f"{col1}_plus_{col2}"
    df[sum_label] = df[col1] + df[col2]

col1 = "open_price"
col2 = "close_price"
calculate_sum(dataset, col1=col1, col2=col2)
"""

PROD_FUNC = """
import pandas as pd

def calculate_product(df, col1='column1', col2='column2'):
    product_label = f"{col1}_times_{col2}"
    df[product_label] = df[col1] * df[col2]

# Usage example:
col1 = "open_price"
col2 = "close_price"
calculate_product(dataset, col1=col1, col2=col2)
"""

DIFF_SQRD_FUNC = """
import pandas as pd

def calculate_difference_squared(df, col1='column1', col2='column2'):
    diff_squared_label = f"{col1}_minus_{col2}_squared"
    df[diff_squared_label] = (df[col1] - df[col2]) ** 2

# Usage example:
col1 = "open_price"
col2 = "close_price"
calculate_difference_squared(dataset, col1=col1, col2=col2)
"""

ROW_WISE_MIN_MAX = """
import pandas as pd

def calculate_min(df, col1='column1', col2='column2'):
    min_label = f"min_of_{col1}_and_{col2}"
    df[min_label] = df[[col1, col2]].min(axis=1)

def calculate_max(df, col1='column1', col2='column2'):
    max_label = f"max_of_{col1}_and_{col2}"
    df[max_label] = df[[col1, col2]].max(axis=1)

# Usage example for Min:
col1_min = "open_price"
col2_min = "close_price"
calculate_min(dataset, col1=col1_min, col2=col2_min)

# Usage example for Max:
col1_max = "open_price"
col2_max = "close_price"
calculate_max(dataset, col1=col1_max, col2=col2_max)
"""


TRIM_TO_NONE = """
import pandas as pd

def trim_initial_values_from_first_non_null(df, column='RSI', num_values=None):
    # Find the index of the first non-null value in the specified column
    first_non_null_index = df[column].first_valid_index()
    if first_non_null_index is not None and num_values is not None:
        # Set the values to None from the first non-null index for the next num_values entries
        end_index = first_non_null_index + num_values
        df.loc[first_non_null_index:end_index, column] = None

# Usage example:
column = 'RSI'
num_values = 14  # Number of initial valid entries to set as None after the first non-null
trim_initial_values_from_first_non_null(dataset, column=column, num_values=num_values)
"""

COLS_CROSSING = """
import pandas as pd

def calculate_crossings(df, column_a='column_a', column_b='column_b'):
    # Create labels for crossings
    crossing_above_label = f"{column_a}_crosses_above_{column_b}"
    crossing_below_label = f"{column_a}_crosses_below_{column_b}"

    # Determine where column_a crosses above column_b
    df[crossing_above_label] = ((df[column_a] >= df[column_b]) & (df[column_a].shift(1) < df[column_b].shift(1))).astype(int)

    # Determine where column_a crosses below column_b
    df[crossing_below_label] = ((df[column_a] <= df[column_b]) & (df[column_a].shift(1) > df[column_b].shift(1))).astype(int)

# Usage example
column_a = "open_price"
column_b = "close_price"
calculate_crossings(dataset, column_a=column_a, column_b=column_b)
"""

SALARY_DAY = """
import pandas as pd

def calculate_second_last_day_of_month(df, date_col='kline_open_time'):
    # Convert the timestamp to datetime format
    df["temp_col"] = pd.to_datetime(df[date_col], unit='ms')
    
    # Calculate the last day of each month in the dataset
    df['month_last_day'] = df["temp_col"] + pd.offsets.MonthEnd(0)
    
    # Calculate the second last day of the month by subtracting one day from the last day
    df['second_last_day_of_month'] = df['month_last_day'] - pd.Timedelta(days=1)
    
    # Create a boolean indicator for the second last day of the month
    df['is_second_last_day_of_month'] = (df["temp_col"].dt.date == df['second_last_day_of_month'].dt.date).astype(int)
    
    # Drop the helper columns
    df.drop(['month_last_day', 'second_last_day_of_month', "temp_col"], axis=1, inplace=True)

# Usage example:
date_col = "kline_open_time"
calculate_second_last_day_of_month(dataset, date_col=date_col)
"""

IS_MONDAY = """
import pandas as pd

def calculate_is_monday(df, timestamp_col='kline_open_time'):
    # Convert timestamp from milliseconds to datetime
    df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
    
    # Determine if the day is Monday (Monday is 0 in dt.weekday)
    df['is_monday'] = (df['datetime'].dt.weekday == 0).astype(int)
    
    # Drop the temporary datetime column
    df.drop(columns=['datetime'], inplace=True)

# Usage example:
timestamp_col = "kline_open_time"
calculate_is_monday(dataset, timestamp_col=timestamp_col)
"""

EVENT_PERC_CHANGE_FLAG = """
import pandas as pd

def create_change_flag(df, close_col='close_price', flag_col='is_monday', change_threshold=2, direction='up'):
    # Create a column for the previous day's closing price
    df['previous_close'] = df[close_col].shift(1)
    
    # Calculate the percentage change from the previous close to the current close
    df['percentage_change'] = ((df[close_col] - df['previous_close']) / df['previous_close']) * 100
    
    # Define the condition based on the direction of the change
    if direction.lower() == 'up':
        condition = df['percentage_change'] > change_threshold
    elif direction.lower() == 'down':
        condition = df['percentage_change'] < -change_threshold
    else:
        raise ValueError("Direction must be 'up' or 'down'")
    
    # Flag days based on the specified flag column and the condition
    flag_label = f"{flag_col}_changed_{direction}_more_than_{change_threshold}_percent"
    df[flag_label] = (condition & (df[flag_col] == 1)).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['previous_close', 'percentage_change'], inplace=True)

# Usage example:
close_col = "close_price"
flag_col = "is_monday"
change_threshold = 2
direction = "up"
create_change_flag(dataset, close_col=close_col, flag_col=flag_col, change_threshold=change_threshold, direction=direction)
"""

INCREASING_STREAK = """
import pandas as pd

def calculate_increasing_streak(df, column='close_price', period=3):
    increasing_streak_label = f"increasing_streak_{period}_{column}"
    
    # Create a helper column to check if current value is greater than previous value
    df['increasing'] = (df[column] > df[column].shift(1)).astype(int)
    
    # Calculate rolling sum of the 'increasing' helper column
    df[increasing_streak_label] = df['increasing'].rolling(window=period).sum()
    
    # Set indicator: 1 if rolling sum equals period (all previous values were increasing), else 0
    df[increasing_streak_label] = (df[increasing_streak_label] == period).astype(int)
    
    # Drop the helper column
    df.drop(columns=['increasing'], inplace=True)

# Usage example:
column = "RSI_30_MA_50_close_price"
period = 3
calculate_increasing_streak(dataset, column=column, period=period)
"""


DECREASING_STREAK = """
import pandas as pd

def calculate_decreasing_streak(df, column='close_price', period=3):
    decreasing_streak_label = f"decreasing_streak_{period}_{column}"
    
    # Create a helper column to check if current value is less than previous value
    df['decreasing'] = (df[column] < df[column].shift(1)).astype(int)
    
    # Calculate rolling sum of the 'decreasing' helper column
    df[decreasing_streak_label] = df['decreasing'].rolling(window=period).sum()
    
    # Set indicator: 1 if rolling sum equals period (all previous values were decreasing), else 0
    df[decreasing_streak_label] = (df[decreasing_streak_label] == period).astype(int)
    
    # Drop the helper column
    df.drop(columns=['decreasing'], inplace=True)

# Usage example:
column = "close_price"
period = 3
calculate_decreasing_streak(dataset, column=column, period=period)
"""

CONDITIONAL_FLAG = """
import pandas as pd
import numpy as np

def create_market_indicator(df, col1='close_price', col2='MA_4032_close_price', threshold=1.2, indicator_label='bull_market_flag'):
    # Safely create the indicator column based on the given condition
    df[indicator_label] = np.where(df[col1] > threshold * df[col2], 1, 0)
    return df

# Define the dependent variables
col1 = 'close_price'
col2 = 'MA_4032_close_price'
threshold = 1.2
indicator_label = 'is_bull_market'

# Create the indicator column
df = create_market_indicator(dataset, col1=col1, col2=col2, threshold=threshold, indicator_label=indicator_label)
"""

ALL_CONDS_TRUE = """
import pandas as pd

def calculate_combined_indicator(df, condition_cols=['cond1', 'cond2'], invert_flags=['bull_market_flag']):
    # Create a copy of the DataFrame to avoid modifying the original
    temp_df = df.copy()
    
    # Initialize the list for final condition column names
    final_condition_cols = condition_cols.copy()
    
    # Invert the specified flags and update the final condition column names
    for flag in invert_flags:
        if flag in temp_df.columns:
            temp_df[flag] = 1 - temp_df[flag]
            final_condition_cols[final_condition_cols.index(flag)] = f"not_{flag}"
    
    combined_label = f"combined_{'_'.join(final_condition_cols)}"
    
    # Generate the combined indicator
    temp_df[combined_label] = temp_df[condition_cols].all(axis=1).astype(int)
    
    # Drop temporary columns if any were added
    df[combined_label] = temp_df[combined_label]

# Usage example:
condition_cols = ['is_thurs_5_utc', 'bull_market_flag']
invert_flags = ['bull_market_flag']
calculate_combined_indicator(dataset, condition_cols=condition_cols, invert_flags=invert_flags)
"""

CONTAINS_TRUE_N_LOOKBACK = """
import pandas as pd

def calculate_indicator(df, source_col='source_column', n=5):
    indicator_label = f"indicator_last_{n}_rows_{source_col}"
    df[indicator_label] = df[source_col].rolling(window=n, min_periods=1).apply(lambda x: 1 if 1 in x.values else 0)

source_col = "source_column"
n = 5
calculate_indicator(dataset, source_col=source_col, n=n)
"""

CDL3BLACKCROWS = """
import pandas as pd

def calculate_three_black_crows(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Initialize the column for the pattern
    df['CDL3BLACKCROWS'] = 0

    # Define the condition for Three Black Crows
    for i in range(2, len(df)):
        if (df[close_col][i] < df[open_col][i] and  # Today's close is lower than today's open
            df[close_col][i-1] < df[open_col][i-1] and  # Yesterday's close was lower than yesterday's open
            df[close_col][i-2] < df[open_col][i-2] and  # Day before yesterday's close was lower than day before yesterday's open
            df[close_col][i] < df[low_col][i-1] and  # Today's close is lower than yesterday's low
            df[close_col][i-1] < df[low_col][i-2]):  # Yesterday's close is lower than day before yesterday's low
            
            df['CDL3BLACKCROWS'][i] = 1

    # Drop any helper columns if added
    # In this case, no helper columns were added

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_three_black_crows(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL3INSIDE = """
import pandas as pd

def calculate_cdl3inside(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Create helper columns for previous two days
    df['prev_close'] = df[close_col].shift(1)
    df['prev_open'] = df[open_col].shift(1)
    df['prev_high'] = df[high_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Identify Bullish and Bearish patterns
    df['bullish_pattern'] = ((df['prev2_close'] > df['prev2_open']) &  # First candle is bullish
                             (df['prev_open'] < df['prev_close']) &    # Second candle is bearish
                             (df[close_col] > df[open_col]) &          # Third candle is bullish
                             (df[open_col] < df['prev_close']) &       # Third candle opens within the second candle
                             (df[close_col] > df['prev_open']))        # Third candle closes above the second candle

    df['bearish_pattern'] = ((df['prev2_close'] < df['prev2_open']) &  # First candle is bearish
                             (df['prev_open'] > df['prev_close']) &    # Second candle is bullish
                             (df[close_col] < df[open_col]) &          # Third candle is bearish
                             (df[open_col] > df['prev_close']) &       # Third candle opens within the second candle
                             (df[close_col] < df['prev_open']))        # Third candle closes below the second candle

    # Combine the patterns into one column
    df['CDL3INSIDE'] = df['bullish_pattern'].astype(int) - df['bearish_pattern'].astype(int)
    
    # Drop helper columns
    df.drop(['prev_close', 'prev_open', 'prev_high', 'prev_low', 
             'prev2_close', 'prev2_open', 'prev2_high', 'prev2_low', 
             'bullish_pattern', 'bearish_pattern'], axis=1, inplace=True)

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdl3inside(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL3LINESTRIKE = """
import pandas as pd

def calculate_cdl3linestrike(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Create helper columns for bullish and bearish conditions
    df['bullish1'] = (df[close_col].shift(3) > df[open_col].shift(3)) & (df[close_col].shift(2) > df[open_col].shift(2)) & (df[close_col].shift(1) > df[open_col].shift(1))
    df['bearish1'] = (df[close_col].shift(3) < df[open_col].shift(3)) & (df[close_col].shift(2) < df[open_col].shift(2)) & (df[close_col].shift(1) < df[open_col].shift(1))

    df['bullish2'] = (df[open_col] < df[close_col].shift(1)) & (df[close_col] > df[open_col].shift(3))
    df['bearish2'] = (df[open_col] > df[close_col].shift(1)) & (df[close_col] < df[open_col].shift(3))
    
    df['three_line_strike_bullish'] = (df['bullish1'] & df['bullish2']).astype(int)
    df['three_line_strike_bearish'] = (df['bearish1'] & df['bearish2']).astype(int)

    # Drop helper columns
    df.drop(columns=['bullish1', 'bearish1', 'bullish2', 'bearish2'], inplace=True)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
calculate_cdl3linestrike(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL3OUTSIDE = """
import pandas as pd

def calculate_cdl3outside(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['bullish_candle'] = df[close_col] > df[open_col]
    df['bearish_candle'] = df[close_col] < df[open_col]

    # Three Outside Up
    df['three_outside_up'] = (
        (df['bearish_candle'].shift(2)) &
        (df['bullish_candle'].shift(1)) &
        (df[close_col].shift(1) > df[open_col].shift(2)) &
        (df[open_col].shift(1) < df[close_col].shift(2)) &
        (df[close_col] > df[high_col].shift(1))
    ).astype(int)

    # Three Outside Down
    df['three_outside_down'] = (
        (df['bullish_candle'].shift(2)) &
        (df['bearish_candle'].shift(1)) &
        (df[close_col].shift(1) < df[open_col].shift(2)) &
        (df[open_col].shift(1) > df[close_col].shift(2)) &
        (df[close_col] < df[low_col].shift(1))
    ).astype(int)

    # Drop helper columns
    df.drop(columns=['bullish_candle', 'bearish_candle'], inplace=True)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
calculate_cdl3outside(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL3STARINSOUTH = """
import pandas as pd

def calculate_cdl3starsinthesouth(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['CDL3STARSINSOUTH'] = 0

    for i in range(2, len(df)):
        first_candle = df.iloc[i - 2]
        second_candle = df.iloc[i - 1]
        third_candle = df.iloc[i]

        if (first_candle[close_col] < first_candle[open_col] and 
            second_candle[close_col] < second_candle[open_col] and 
            third_candle[close_col] > third_candle[open_col] and 
            third_candle[low_col] < second_candle[low_col] < first_candle[low_col] and 
            third_candle[high_col] < second_candle[high_col] < first_candle[high_col]):
            df.at[i, 'CDL3STARSINSOUTH'] = 1

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
calculate_cdl3starsinthesouth(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDL3WHITESOLDIERS = """
def calculate_cdl3whitesoldiers(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['white_soldier1'] = ((df[close_col] > df[open_col]) &
                            (df[close_col].shift(1) > df[open_col].shift(1)) &
                            (df[close_col].shift(2) > df[open_col].shift(2)) &
                            (df[open_col] > df[close_col].shift(1)) &
                            (df[open_col].shift(1) > df[close_col].shift(2)) &
                            ((df[close_col] - df[open_col]) > (df[close_col].shift(1) - df[open_col].shift(1))) &
                            ((df[close_col].shift(1) - df[open_col].shift(1)) > (df[close_col].shift(2) - df[open_col].shift(2))))

    df['CDL3WHITESOLDIERS'] = df['white_soldier1'].astype(int)

    # Drop the helper column
    df.drop(columns=['white_soldier1'], inplace=True)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
calculate_cdl3whitesoldiers(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLABANDONEDBABY = """
import pandas as pd

def calculate_abandoned_baby(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Abandoned Baby: Bullish and Bearish
    df['prev_close'] = df[close_col].shift(1)
    df['prev_open'] = df[open_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    df['prev_high'] = df[high_col].shift(1)
    
    df['next_open'] = df[open_col].shift(-1)
    df['next_close'] = df[close_col].shift(-1)
    
    df['is_bullish_abandoned_baby'] = (
        (df['prev_close'] < df['prev_open']) & 
        (df[open_col] > df['prev_close']) & 
        (df[open_col] < df['prev_open']) & 
        (df[close_col] > df[open_col]) & 
        (df['next_open'] > df[close_col]) &
        (df['next_open'] < df[high_col]) & 
        (df['next_close'] > df['next_open'])
    ).astype(int)
    
    df['is_bearish_abandoned_baby'] = (
        (df['prev_close'] > df['prev_open']) & 
        (df[open_col] < df['prev_close']) & 
        (df[open_col] > df['prev_open']) & 
        (df[close_col] < df[open_col]) & 
        (df['next_open'] < df[close_col]) &
        (df['next_open'] > df[low_col]) & 
        (df['next_close'] < df['next_open'])
    ).astype(int)
    
    df.drop(['prev_close', 'prev_open', 'prev_low', 'prev_high', 'next_open', 'next_close'], axis=1, inplace=True)

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_abandoned_baby(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLADVANCEDBLOCK = """
import pandas as pd

def calculate_advance_block(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    # Calculate the candle bodies
    body_1 = df[close_col].shift(3) - df[open_col].shift(3)
    body_2 = df[close_col].shift(2) - df[open_col].shift(2)
    body_3 = df[close_col].shift(1) - df[open_col].shift(1)

    # Determine if all three candles are bullish
    bullish_1 = body_1 > 0
    bullish_2 = body_2 > 0
    bullish_3 = body_3 > 0

    # Determine if each candle opens within the previous candle's body
    open_within_2 = df[open_col].shift(2) < df[close_col].shift(3)
    open_within_3 = df[open_col].shift(1) < df[close_col].shift(2)

    # Determine if each candle's body is smaller than the previous one
    smaller_body_2 = body_2 < body_1
    smaller_body_3 = body_3 < body_2

    # Calculate the advance block condition
    advance_block = (bullish_1 & bullish_2 & bullish_3 &
                     open_within_2 & open_within_3 &
                     smaller_body_2 & smaller_body_3)

    # Add the result to the dataframe
    df['CDL_ADVANCE_BLOCK'] = advance_block.astype(int)

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_advance_block(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLBELTHOLD = """
import pandas as pd

def calculate_belthold(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    # Create helper columns
    df['bullish'] = (df[close_col] > df[open_col]) & (df[open_col] == df[low_col])
    df['bearish'] = (df[close_col] < df[open_col]) & (df[open_col] == df[high_col])
    
    # Assign Belt-hold pattern labels
    df['Bullish_Belt_Hold'] = df['bullish'].astype(int)
    df['Bearish_Belt_Hold'] = df['bearish'].astype(int)
    
    # Drop helper columns
    df.drop(columns=['bullish', 'bearish'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_belthold(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLHIKKAKE = """
import pandas as pd

def calculate_hikkake_pattern(df, high_col='high_price', low_col='low_price', open_col='open_price', close_col='close_price'):
    df['inside_bar'] = ((df[high_col].shift(1) > df[high_col]) & (df[low_col].shift(1) < df[low_col])).astype(int)
    
    df['hikkake_pattern'] = 0
    for i in range(2, len(df)):
        if df['inside_bar'].iloc[i-1] == 1:
            if (df[high_col].iloc[i] > df[high_col].iloc[i-1]) and (df[low_col].iloc[i] < df[low_col].iloc[i-1]):
                df['hikkake_pattern'].iloc[i] = 1

    df.drop(columns=['inside_bar'], inplace=True)

# Usage example:
high_col = "high_price"
low_col = "low_price"
open_col = "open_price"
close_col = "close_price"
calculate_hikkake_pattern(dataset, high_col=high_col, low_col=low_col, open_col=open_col, close_col=close_col)
"""

CDLPIERCING = """
import pandas as pd

def calculate_piercing_pattern(df, open_col='open_price', close_col='close_price', low_col='low_price', high_col='high_price'):
    df['piercing_pattern'] = 0
    
    for i in range(1, len(df)):
        # Check if there is a bearish candle followed by a bullish candle
        if df[close_col].iloc[i-1] < df[open_col].iloc[i-1]:  # Previous day is bearish
            if df[close_col].iloc[i] > df[open_col].iloc[i]:  # Current day is bullish
                # Check if the bullish candle opens below the previous day's low
                if df[open_col].iloc[i] < df[low_col].iloc[i-1]:
                    # Check if the bullish candle closes above the midpoint of the previous day's body
                    prev_midpoint = (df[open_col].iloc[i-1] + df[close_col].iloc[i-1]) / 2
                    if df[close_col].iloc[i] > prev_midpoint:
                        df['piercing_pattern'].iloc[i] = 1

# Usage example:
open_col = "open_price"
close_col = "close_price"
low_col = "low_price"
high_col = "high_price"
calculate_piercing_pattern(dataset, open_col=open_col, close_col=close_col, low_col=low_col, high_col=high_col)
"""

CDLBREAKAWAY = """
def calculate_cdlbreakaway(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['Breakaway'] = 0

    for i in range(4, len(df)):
        if (
            df[open_col].iloc[i-4] < df[close_col].iloc[i-4] and  # Bullish day 1
            df[open_col].iloc[i-3] > df[close_col].iloc[i-3] and  # Bearish day 2
            df[open_col].iloc[i-2] > df[close_col].iloc[i-2] and  # Bearish day 3
            df[open_col].iloc[i-1] > df[close_col].iloc[i-1] and  # Bearish day 4
            df[close_col].iloc[i] > df[open_col].iloc[i] and      # Bullish day 5
            df[close_col].iloc[i] > df[open_col].iloc[i-4]        # Close of day 5 above open of day 1
        ):
            df.at[i, 'Breakaway'] = 1  # Bullish Breakaway

        elif (
            df[open_col].iloc[i-4] > df[close_col].iloc[i-4] and  # Bearish day 1
            df[open_col].iloc[i-3] < df[close_col].iloc[i-3] and  # Bullish day 2
            df[open_col].iloc[i-2] < df[close_col].iloc[i-2] and  # Bullish day 3
            df[open_col].iloc[i-1] < df[close_col].iloc[i-1] and  # Bullish day 4
            df[close_col].iloc[i] < df[open_col].iloc[i] and      # Bearish day 5
            df[close_col].iloc[i] < df[open_col].iloc[i-4]        # Close of day 5 below open of day 1
        ):
            df.at[i, 'Breakaway'] = -1  # Bearish Breakaway

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdlbreakaway(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLCLOSINGMARUBOZU = """
def calculate_cdlclosingmarubozu(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['Closing_Marubozu'] = 0

    for i in range(len(df)):
        if (
            df[close_col].iloc[i] == df[high_col].iloc[i] and  # Closing price equals the high price
            df[open_col].iloc[i] < df[close_col].iloc[i]       # Open price is less than the closing price
        ):
            df.at[i, 'Closing_Marubozu'] = 1  # Bullish Closing Marubozu

        elif (
            df[close_col].iloc[i] == df[low_col].iloc[i] and   # Closing price equals the low price
            df[open_col].iloc[i] > df[close_col].iloc[i]       # Open price is greater than the closing price
        ):
            df.at[i, 'Closing_Marubozu'] = -1  # Bearish Closing Marubozu

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdlclosingmarubozu(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLCONCEALBABYSWALL = """
def calculate_cdlconcealbabyswall(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['Concealing_Baby_Swallow'] = 0

    for i in range(3, len(df)):
        if (
            df[close_col].iloc[i-3] < df[open_col].iloc[i-3] and      # First candle is bearish
            df[close_col].iloc[i-2] < df[open_col].iloc[i-2] and      # Second candle is bearish
            df[low_col].iloc[i-2] < df[low_col].iloc[i-3] and         # Second candle's low is lower than the first candle's low
            df[close_col].iloc[i-1] < df[open_col].iloc[i-1] and      # Third candle is bearish
            df[low_col].iloc[i-1] < df[low_col].iloc[i-2] and         # Third candle's low is lower than the second candle's low
            df[close_col].iloc[i] > df[open_col].iloc[i] and          # Fourth candle is bullish
            df[close_col].iloc[i] > df[high_col].iloc[i-1] and        # Fourth candle closes above the high of the third candle
            df[open_col].iloc[i] < df[close_col].iloc[i-1]            # Fourth candle opens below the close of the third candle
        ):
            df.at[i, 'Concealing_Baby_Swallow'] = 1  # Concealing Baby Swallow pattern

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdlconcealbabyswall(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLCOUNTERATTACK = """
def calculate_cdlcounterattack(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['Counterattack'] = 0

    for i in range(1, len(df)):
        # Bullish Counterattack
        if (
            df[close_col].iloc[i-1] < df[open_col].iloc[i-1] and  # Previous candle is bearish
            df[close_col].iloc[i] > df[open_col].iloc[i] and      # Current candle is bullish
            df[close_col].iloc[i] == df[close_col].iloc[i-1]      # Closing prices are the same
        ):
            df.at[i, 'Counterattack'] = 1  # Bullish Counterattack

        # Bearish Counterattack
        elif (
            df[close_col].iloc[i-1] > df[open_col].iloc[i-1] and  # Previous candle is bullish
            df[close_col].iloc[i] < df[open_col].iloc[i] and      # Current candle is bearish
            df[close_col].iloc[i] == df[close_col].iloc[i-1]      # Closing prices are the same
        ):
            df.at[i, 'Counterattack'] = -1  # Bearish Counterattack

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdlcounterattack(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

CDLDARKCLOUDCOVER = """
def calculate_cdldarkcloudcover(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price'):
    df['Dark_Cloud_Cover'] = 0

    for i in range(1, len(df)):
        if (
            df[close_col].iloc[i-1] > df[open_col].iloc[i-1] and  # Previous candle is bullish
            df[open_col].iloc[i] > df[high_col].iloc[i-1] and    # Current candle opens above the high of the previous candle
            df[close_col].iloc[i] < df[open_col].iloc[i] and     # Current candle is bearish
            df[close_col].iloc[i] < (df[open_col].iloc[i] + df[close_col].iloc[i-1]) / 2 and  # Closes below the midpoint of the previous candle
            df[close_col].iloc[i] > df[open_col].iloc[i-1]       # Closes above the open of the previous candle
        ):
            df.at[i, 'Dark_Cloud_Cover'] = 1  # Dark Cloud Cover pattern

# Usage example:
open_col = "open_price"
high_col = "high_price"
low_col = "low_price"
close_col = "close_price"
calculate_cdldarkcloudcover(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
"""

IS_BEAR_MARKET = """
import pandas as pd

def calculate_bear_market_indicator(df, column='close_price', sma_column='SMA_50_close_price', coefficient=0.8):
    bear_market_label = f"IS_BEAR_MARKET_{column}_coeff_{coefficient}"
    df[bear_market_label] = (df[column] < df[sma_column] * coefficient).astype(int)

# Usage example:
column = 'close_price'
sma_column = 'MA_50_close_price'
coefficient = 0.8
calculate_bear_market_indicator(dataset, column=column, sma_column=sma_column, coefficient=coefficient)
"""

CDLDOJI = """
import pandas as pd

def calculate_doji(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', threshold=0.001):
    doji_label = 'CDLDOJI'
    df[doji_label] = ((abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * threshold))).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
threshold = 0.001
calculate_doji(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, threshold=threshold)
"""

CDLDOJISTAR = """
import pandas as pd

def calculate_doji_star(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', threshold=0.001):
    doji_star_label = 'CDLDOJISTAR'
    
    df['doji'] = ((abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * threshold))).astype(int)
    
    df['prev_close'] = df[close_col].shift(1)
    df['prev_open'] = df[open_col].shift(1)
    
    df[doji_star_label] = ((df['doji'] == 1) & 
                           (df['prev_close'] > df['prev_open']) & 
                           (df[close_col] < df[open_col])).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['doji', 'prev_close', 'prev_open'], inplace=True)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
threshold = 0.001
calculate_doji_star(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, threshold=threshold)
"""

CDLDRAGONFLYDOJI = """
import pandas as pd

def calculate_dragonfly_doji(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', threshold=0.001):
    dragonfly_doji_label = 'CDLDRAGONFLYDOJI'
    
    df[dragonfly_doji_label] = (
        (abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * threshold)) & 
        ((df[high_col] - df[open_col]) <= (threshold * (df[high_col] - df[low_col]))) & 
        ((df[high_col] - df[close_col]) <= (threshold * (df[high_col] - df[low_col])))
    ).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
threshold = 0.001
calculate_dragonfly_doji(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, threshold=threshold)
"""

CDLENGULFING = """
import pandas as pd

def calculate_engulfing(df, open_col='open_price', close_col='close_price'):
    engulfing_label = 'CDLENGULFING'
    
    # Shift the open and close prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    
    # Bullish Engulfing: Previous close is less than previous open, current close is greater than current open,
    # and current open is less than previous close, and current close is greater than previous open.
    bullish_engulfing = ((df['prev_close'] < df['prev_open']) & 
                         (df[close_col] > df[open_col]) & 
                         (df[open_col] < df['prev_close']) & 
                         (df[close_col] > df['prev_open'])).astype(int)
    
    # Bearish Engulfing: Previous close is greater than previous open, current close is less than current open,
    # and current open is greater than previous close, and current close is less than previous open.
    bearish_engulfing = ((df['prev_close'] > df['prev_open']) & 
                         (df[close_col] < df[open_col]) & 
                         (df[open_col] > df['prev_close']) & 
                         (df[close_col] < df['prev_open'])).astype(int)
    
    df[engulfing_label] = bullish_engulfing - bearish_engulfing
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_engulfing(dataset, open_col=open_col, close_col=close_col)
"""
CDLEVENINGDOJISTAR = """
import pandas as pd

def calculate_evening_doji_star(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    evening_doji_star_label = 'CDLEVENINGDOJISTAR'
    
    # Identify Doji candles
    df['doji'] = ((abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * threshold))).astype(int)
    
    # Identify potential Evening Doji Star patterns
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    
    # Evening Doji Star: 
    # - The first candle is a long bullish candle.
    # - The second candle is a Doji that gaps above the first candle.
    # - The third candle is a long bearish candle that closes below the midpoint of the first candle.
    long_bullish_candle = (df['prev2_close'] > df['prev2_open'])
    long_bearish_candle = (df[close_col] < df[open_col])
    doji_gapped_above = (df['doji'] == 1) & (df[open_col] > df['prev_close']) & (df[close_col] > df['prev_close'])
    bearish_close_below_midpoint = (df[close_col] < (df['prev2_open'] + df['prev2_close']) / 2)
    
    df[evening_doji_star_label] = (long_bullish_candle & doji_gapped_above & long_bearish_candle & bearish_close_below_midpoint).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['doji', 'prev_open', 'prev_close', 'prev2_open', 'prev2_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_evening_doji_star(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""
CDLEVENINGSTAR = """
import pandas as pd

def calculate_evening_star(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    evening_star_label = 'CDLEVENINGSTAR'
    
    # Identify potential Evening Star patterns
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    
    # Evening Star: 
    # - The first candle is a long bullish candle.
    # - The second candle has a small real body that gaps above the first candle.
    # - The third candle is a long bearish candle that closes below the midpoint of the first candle.
    long_bullish_candle = (df['prev2_close'] > df['prev2_open'])
    small_body_candle = (abs(df['prev_close'] - df['prev_open']) <= (df[high_col] - df[low_col]) * threshold)
    long_bearish_candle = (df[close_col] < df[open_col])
    small_body_gapped_above = (df['prev_open'] > df['prev2_close']) & (df['prev_close'] > df['prev2_close'])
    bearish_close_below_midpoint = (df[close_col] < (df['prev2_open'] + df['prev2_close']) / 2)
    
    df[evening_star_label] = (long_bullish_candle & small_body_candle & small_body_gapped_above & long_bearish_candle & bearish_close_below_midpoint).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev2_open', 'prev2_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_evening_star(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""

CDLGAPSIDESIDEWHITE = """
import pandas as pd

def calculate_gap_side_by_side_white(df, open_col='open_price', close_col='close_price'):
    gap_side_by_side_white_label = 'CDLGAPSIDESIDEWHITE'
    
    # Previous two candles' open and close prices
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    
    # Up-gap side-by-side white lines
    up_gap = (df[open_col] > df['prev_close']) & (df['prev_open'] > df['prev2_close'])
    white_lines = (df[close_col] > df[open_col]) & (df['prev_close'] > df['prev_open']) & (df['prev2_close'] > df['prev2_open'])
    
    # Down-gap side-by-side white lines
    down_gap = (df[open_col] < df['prev_close']) & (df['prev_open'] < df['prev2_close'])
    
    df[gap_side_by_side_white_label] = ((up_gap & white_lines) | (down_gap & white_lines)).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev2_open', 'prev2_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_gap_side_by_side_white(dataset, open_col=open_col, close_col=close_col)
"""

CDLGRAVESTONEDOJI = """
import pandas as pd

def calculate_gravestone_doji(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', threshold=0.001):
    gravestone_doji_label = 'CDLGRAVESTONEDOJI'
    
    df[gravestone_doji_label] = (
        (abs(df[close_col] - df[open_col]) <= (df[high_col] - df[low_col]) * threshold) & 
        (abs(df[close_col] - df[high_col]) <= (df[high_col] - df[low_col]) * threshold) & 
        (abs(df[open_col] - df[high_col]) <= (df[high_col] - df[low_col]) * threshold)
    ).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
threshold = 0.001
calculate_gravestone_doji(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, threshold=threshold)
"""

CDLHAMMER = """
import pandas as pd

def calculate_hammer(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', body_threshold=0.001, shadow_ratio=2):
    hammer_label = 'CDLHAMMER'
    
    # Calculate the real body size and the lower shadow size
    real_body = abs(df[close_col] - df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    
    df[hammer_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) & 
        (lower_shadow >= real_body * shadow_ratio) &
        (df[close_col] > df[open_col])  # To confirm it's a bullish candle
    ).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
body_threshold = 0.001
shadow_ratio = 2
calculate_hammer(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLHANGINGMAN = """
import pandas as pd

def calculate_hanging_man(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', body_threshold=0.001, shadow_ratio=2):
    hanging_man_label = 'CDLHANGINGMAN'
    
    # Calculate the real body size and the lower shadow size
    real_body = abs(df[close_col] - df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    
    df[hanging_man_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) & 
        (lower_shadow >= real_body * shadow_ratio) & 
        (df[close_col] < df[open_col])  # To confirm it's a bearish candle
    ).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
body_threshold = 0.001
shadow_ratio = 2
calculate_hanging_man(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLHARAMI = """
import pandas as pd

def calculate_harami(df, open_col='open_price', close_col='close_price'):
    harami_label = 'CDLHARAMI'
    
    # Shift the open and close prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    
    # Bullish Harami: Previous candle is bearish, current candle is bullish, and the current candle's body is within the previous candle's body
    bullish_harami = (
        (df['prev_close'] < df['prev_open']) &  # Previous candle is bearish
        (df[close_col] > df[open_col]) &  # Current candle is bullish
        (df[open_col] > df['prev_close']) & 
        (df[close_col] < df['prev_open'])
    ).astype(int)
    
    # Bearish Harami: Previous candle is bullish, current candle is bearish, and the current candle's body is within the previous candle's body
    bearish_harami = (
        (df['prev_close'] > df['prev_open']) &  # Previous candle is bullish
        (df[close_col] < df[open_col]) &  # Current candle is bearish
        (df[open_col] < df['prev_close']) & 
        (df[close_col] > df['prev_open'])
    ).astype(int)
    
    df[harami_label] = bullish_harami - bearish_harami
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_harami(dataset, open_col=open_col, close_col=close_col)
"""

CDLHARAMICROSS = """
import pandas as pd

def calculate_harami_cross(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    harami_cross_label = 'CDLHARAMICROSS'
    
    # Identify Doji candles
    df['doji'] = ((abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * threshold))).astype(int)
    
    # Shift the open, close, and doji columns to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['doji_shift'] = df['doji'].shift(1)
    
    # Bullish Harami Cross: Previous candle is bearish, current candle is a Doji, and the Doji's body is within the previous candle's body
    bullish_harami_cross = (
        (df['prev_close'] < df['prev_open']) &  # Previous candle is bearish
        (df['doji'] == 1) &  # Current candle is Doji
        (df[open_col] > df['prev_close']) & 
        (df[close_col] < df['prev_open'])
    ).astype(int)
    
    # Bearish Harami Cross: Previous candle is bullish, current candle is a Doji, and the Doji's body is within the previous candle's body
    bearish_harami_cross = (
        (df['prev_close'] > df['prev_open']) &  # Previous candle is bullish
        (df['doji'] == 1) &  # Current candle is Doji
        (df[open_col] < df['prev_close']) & 
        (df[close_col] > df['prev_open'])
    ).astype(int)
    
    df[harami_cross_label] = bullish_harami_cross - bearish_harami_cross
    
    # Drop the helper columns
    df.drop(columns=['doji', 'prev_open', 'prev_close', 'doji_shift'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_harami_cross(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""

CDLHIGHWAVE = """
import pandas as pd

def calculate_high_wave(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_threshold=0.001, shadow_ratio=2):
    high_wave_label = 'CDLHIGHWAVE'
    
    # Calculate the real body size
    real_body = abs(df[close_col] - df[open_col])
    # Calculate the upper and lower shadows
    upper_shadow = df[high_col] - df[close_col].where(df[close_col] > df[open_col], df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    
    df[high_wave_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) & 
        (upper_shadow >= real_body * shadow_ratio) & 
        (lower_shadow >= real_body * shadow_ratio)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_threshold = 0.001
shadow_ratio = 2
calculate_high_wave(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLHIKKAKEMOD = """
import pandas as pd

def calculate_modified_hikkake(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    modified_hikkake_label = 'CDLHIKKAKEMOD'
    
    # Previous two days' high and low prices
    df['prev_high'] = df[high_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Inside Day: Current high is less than previous high and current low is higher than previous low
    inside_day = (df[high_col] < df['prev_high']) & (df[low_col] > df['prev_low'])
    
    # Modified Hikkake pattern:
    # - The second day (previous day) has a high lower than the first day (two days ago) and a low higher than the first day
    # - The current day breaks the previous day's range
    hikkake_down = (df['prev_high'] < df['prev2_high']) & (df['prev_low'] > df['prev2_low']) & (df[close_col] < df['prev_low'])
    hikkake_up = (df['prev_high'] < df['prev2_high']) & (df['prev_low'] > df['prev2_low']) & (df[close_col] > df['prev_high'])
    
    df[modified_hikkake_label] = (inside_day & (hikkake_down | hikkake_up)).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_high', 'prev_low', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_modified_hikkake(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLHOMINGPIGEON = """
import pandas as pd

def calculate_homing_pigeon(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    homing_pigeon_label = 'CDLHOMINGPIGEON'
    
    # Shift the open and close prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    
    # Homing Pigeon:
    # - The previous candle is bearish
    # - The current candle is bearish
    # - The current candle's body is within the previous candle's body
    previous_bearish = df['prev_close'] < df['prev_open']
    current_bearish = df[close_col] < df[open_col]
    body_within_previous = (df[open_col] >= df['prev_close']) & (df[close_col] <= df['prev_open'])
    
    df[homing_pigeon_label] = (previous_bearish & current_bearish & body_within_previous).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_homing_pigeon(dataset, open_col=open_col, close_col=close_col)
"""

CDLIDENTICAL3CROWS = """
import pandas as pd

def calculate_identical_three_crows(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    identical_three_crows_label = 'CDLIDENTICAL3CROWS'
    
    # Shift the open and close prices to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    
    # Identical Three Crows:
    # - Three consecutive long bearish candles
    # - Each opens within the previous candle's real body
    # - Each closes lower than the previous candle's close
    first_bearish = df['prev2_close'] < df['prev2_open']
    second_bearish = df['prev1_close'] < df['prev1_open']
    third_bearish = df[close_col] < df[open_col]
    
    second_within_first = (df['prev1_open'] <= df['prev2_close']) & (df['prev1_open'] >= df['prev2_open'])
    third_within_second = (df[open_col] <= df['prev1_close']) & (df[open_col] >= df['prev1_open'])
    
    each_lower_close = (df['prev1_close'] < df['prev2_close']) & (df[close_col] < df['prev1_close'])
    
    df[identical_three_crows_label] = (first_bearish & second_bearish & third_bearish & second_within_first & third_within_second & each_lower_close).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev2_open', 'prev2_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_identical_three_crows(dataset, open_col=open_col, close_col=close_col)
"""
CDLINNECK = """
import pandas as pd

def calculate_in_neck(df, open_col='open_price', close_col='close_price', low_col='low_price'):
    in_neck_label = 'CDLINNECK'
    
    # Shift the open and close prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    # In-Neck Pattern:
    # - The previous candle is a long bearish candle.
    # - The current candle opens below the previous candle's low.
    # - The current candle closes slightly above or at the previous candle's close.
    previous_long_bearish = df['prev_close'] < df['prev_open']
    current_opens_below_prev_low = df[open_col] < df['prev_low']
    current_closes_near_prev_close = (df[close_col] >= df['prev_close']) & (df[close_col] <= df['prev_open'])
    
    df[in_neck_label] = (previous_long_bearish & current_opens_below_prev_low & current_closes_near_prev_close).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
low_col = 'low_price'
calculate_in_neck(dataset, open_col=open_col, close_col=close_col, low_col=low_col)
"""

CDLINVERTEDHAMMER = """
import pandas as pd

def calculate_inverted_hammer(df, open_col='open_price', high_col='high_price', low_col='low_price', close_col='close_price', body_threshold=0.001, shadow_ratio=2):
    inverted_hammer_label = 'CDLINVERTEDHAMMER'
    
    # Calculate the real body size and the upper shadow size
    real_body = abs(df[close_col] - df[open_col])
    upper_shadow = df[high_col] - df[close_col].where(df[close_col] > df[open_col], df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    
    df[inverted_hammer_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) & 
        (upper_shadow >= real_body * shadow_ratio) & 
        (lower_shadow <= real_body)
    ).astype(int)

# Usage example:
open_col = 'open_price'
high_col = 'high_price'
low_col = 'low_price'
close_col = 'close_price'
body_threshold = 0.001
shadow_ratio = 2
calculate_inverted_hammer(dataset, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLKICKING = """
import pandas as pd

def calculate_kicking(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    kicking_label = 'CDLKICKING'
    
    # Shift the open and close prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev_high'] = df[high_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    # Kicking Pattern:
    # - The previous candle is a marubozu (either bullish or bearish).
    # - The current candle is a marubozu of the opposite color.
    prev_marubozu_bullish = (df['prev_open'] == df['prev_low']) & (df['prev_close'] == df['prev_high'])
    prev_marubozu_bearish = (df['prev_open'] == df['prev_high']) & (df['prev_close'] == df['prev_low'])
    
    curr_marubozu_bullish = (df[open_col] == df[low_col]) & (df[close_col] == df[high_col])
    curr_marubozu_bearish = (df[open_col] == df[high_col]) & (df[close_col] == df[low_col])
    
    kicking_bullish = prev_marubozu_bearish & curr_marubozu_bullish
    kicking_bearish = prev_marubozu_bullish & curr_marubozu_bearish
    
    df[kicking_label] = (kicking_bullish | kicking_bearish).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev_high', 'prev_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_kicking(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLKICKINGBYLENGTH = """
import pandas as pd

def calculate_kicking_by_length(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    kicking_by_length_label = 'CDLKICKINGBYLENGTH'
    
    # Shift the open, close, high, and low prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev_high'] = df[high_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    # Determine if the previous and current candles are marubozu
    prev_marubozu_bullish = (df['prev_open'] == df['prev_low']) & (df['prev_close'] == df['prev_high'])
    prev_marubozu_bearish = (df['prev_open'] == df['prev_high']) & (df['prev_close'] == df['prev_low'])
    
    curr_marubozu_bullish = (df[open_col] == df[low_col]) & (df[close_col] == df[high_col])
    curr_marubozu_bearish = (df[open_col] == df[high_col]) & (df[close_col] == df[low_col])
    
    # Calculate the lengths of the previous and current marubozu candles
    prev_length = abs(df['prev_close'] - df['prev_open'])
    curr_length = abs(df[close_col] - df[open_col])
    
    # Determine the kicking pattern based on the longer marubozu
    kicking_bullish = (prev_marubozu_bearish & curr_marubozu_bullish & (curr_length > prev_length)).astype(int)
    kicking_bearish = (prev_marubozu_bullish & curr_marubozu_bearish & (curr_length > prev_length)).astype(int)
    
    df[kicking_by_length_label] = kicking_bullish - kicking_bearish
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev_high', 'prev_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_kicking_by_length(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLLADDERBOTTOM = """
import pandas as pd

def calculate_ladder_bottom(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    ladder_bottom_label = 'CDLLADDERBOTTOM'
    
    # Shift the open, close, high, and low prices to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    df['prev3_open'] = df[open_col].shift(3)
    df['prev3_close'] = df[close_col].shift(3)
    df['prev3_high'] = df[high_col].shift(3)
    df['prev3_low'] = df[low_col].shift(3)
    
    df['prev4_open'] = df[open_col].shift(4)
    df['prev4_close'] = df[close_col].shift(4)
    df['prev4_high'] = df[high_col].shift(4)
    df['prev4_low'] = df[low_col].shift(4)
    
    # Ladder Bottom Pattern:
    # - The first three candles are long bearish candles with consecutively lower closes.
    # - The fourth candle is a small bearish or bullish candle with a lower low and a higher close.
    # - The fifth candle is a long bullish candle that closes above the fourth candle's high.
    first_three_bearish = (df['prev4_close'] < df['prev4_open']) & (df['prev3_close'] < df['prev3_open']) & (df['prev2_close'] < df['prev2_open'])
    consecutively_lower_closes = (df['prev3_close'] < df['prev4_close']) & (df['prev2_close'] < df['prev3_close'])
    fourth_candle_small = abs(df['prev1_close'] - df['prev1_open']) < abs(df['prev2_close'] - df['prev2_open'])
    fourth_lower_low_higher_close = (df['prev1_low'] < df['prev2_low']) & (df['prev1_close'] > df['prev2_close'])
    fifth_long_bullish = (df[close_col] > df[open_col]) & (df[close_col] > df['prev1_high'])
    
    df[ladder_bottom_label] = (first_three_bearish & consecutively_lower_closes & fourth_candle_small & fourth_lower_low_higher_clos
"""

CDLLONGLEGGEDDOJI = """
import pandas as pd

def calculate_long_legged_doji(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_threshold=0.001):
    long_legged_doji_label = 'CDLLONGLEGGEDDOJI'
    
    # Calculate the real body size
    real_body = abs(df[close_col] - df[open_col])
    
    # Long Legged Doji:
    # - Very small real body
    # - Long upper and lower shadows
    df[long_legged_doji_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) &
        (abs(df[high_col] - df[open_col]) > real_body * 2) &
        (abs(df[low_col] - df[open_col]) > real_body * 2)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_threshold = 0.001
calculate_long_legged_doji(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_threshold=body_threshold)
"""

CDLLONGLINE = """
import pandas as pd

def calculate_long_line_candle(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_ratio=0.7):
    long_line_candle_label = 'CDLLONGLINE'
    
    # Calculate the real body size
    real_body = abs(df[close_col] - df[open_col])
    
    # Calculate the range (high - low)
    candle_range = df[high_col] - df[low_col]
    
    # Long Line Candle:
    # - The real body is large relative to the range of the candle
    df[long_line_candle_label] = (real_body >= candle_range * body_ratio).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_ratio = 0.7
calculate_long_line_candle(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_ratio=body_ratio)
"""

CDLMATCHINGLOW = """
import pandas as pd

def calculate_matching_low(df, close_col='close_price'):
    matching_low_label = 'CDLMATCHINGLOW'
    
    # Shift the close prices to compare with the previous day
    df['prev_close'] = df[close_col].shift(1)
    
    # Matching Low:
    # - Two consecutive candles with the same close price
    # - Typically occurs in a downtrend
    matching_low = (df[close_col] == df['prev_close']).astype(int)
    
    df[matching_low_label] = matching_low
    
    # Drop the helper columns
    df.drop(columns=['prev_close'], inplace=True)

# Usage example:
close_col = 'close_price'
calculate_matching_low(dataset, close_col=close_col)
"""

CDLMATHOLD = """
import pandas as pd

def calculate_mat_hold(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    mat_hold_label = 'CDLMATHOLD'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    df['prev3_open'] = df[open_col].shift(3)
    df['prev3_close'] = df[close_col].shift(3)
    df['prev3_high'] = df[high_col].shift(3)
    df['prev3_low'] = df[low_col].shift(3)
    
    df['prev4_open'] = df[open_col].shift(4)
    df['prev4_close'] = df[close_col].shift(4)
    df['prev4_high'] = df[high_col].shift(4)
    df['prev4_low'] = df[low_col].shift(4)
    
    # Mat Hold Pattern:
    # - The first candle is a long bullish candle.
    # - The next three candles are small candles that consolidate, with each candle having a higher low.
    # - The fifth candle is a long bullish candle that closes above the high of the first candle.
    first_bullish = df['prev4_close'] > df['prev4_open']
    small_candle_1 = (df['prev3_close'] - df['prev3_open']).abs() < (df['prev4_close'] - df['prev4_open']).abs() * 0.5
    small_candle_2 = (df['prev2_close'] - df['prev2_open']).abs() < (df['prev4_close'] - df['prev4_open']).abs() * 0.5
    small_candle_3 = (df['prev1_close'] - df['prev1_open']).abs() < (df['prev4_close'] - df['prev4_open']).abs() * 0.5
    higher_lows = (df['prev3_low'] > df['prev4_low']) & (df['prev2_low'] > df['prev3_low']) & (df['prev1_low'] > df['prev2_low'])
    fifth_bullish = df[close_col] > df[open_col]
    fifth_closes_above_first = df[close_col] > df['prev4_high']
    
    df[mat_hold_label] = (first_bullish & small_candle_1 & small_candle_2 & small_candle_3 & higher_lows & fifth_bullish & fifth_closes_above_first).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low', 
                     'prev3_open', 'prev3_close', 'prev3_high', 'prev3_low', 
                     'prev4_open', 'prev4_close', 'prev4_high', 'prev4_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_mat_hold(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLMORNINGDOJISTAR = """
import pandas as pd

def calculate_morning_doji_star(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    morning_doji_star_label = 'CDLMORNINGDOJISTAR'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Morning Doji Star Pattern:
    # - The first candle is a long bearish candle.
    # - The second candle is a Doji that gaps below the first candle.
    # - The third candle is a long bullish candle that closes above the midpoint of the first candle's body.
    first_bearish = df['prev2_close'] < df['prev2_open']
    second_doji = (abs(df['prev1_close'] - df['prev1_open']) <= ((df['prev1_high'] - df['prev1_low']) * threshold))
    doji_gapped_down = (df['prev1_low'] < df['prev2_low']) & (df['prev1_high'] < df['prev2_low'])
    third_bullish = df[close_col] > df[open_col]
    third_closes_above_midpoint = df[close_col] > ((df['prev2_open'] + df['prev2_close']) / 2)
    
    df[morning_doji_star_label] = (first_bearish & second_doji & doji_gapped_down & third_bullish & third_closes_above_midpoint).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_morning_doji_star(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""

CDLMORNINGSTAR = """
import pandas as pd

def calculate_morning_star(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    morning_star_label = 'CDLMORNINGSTAR'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Morning Star Pattern:
    # - The first candle is a long bearish candle.
    # - The second candle is a small-bodied candle that gaps below the first candle.
    # - The third candle is a long bullish candle that closes above the midpoint of the first candle's body.
    first_bearish = df['prev2_close'] < df['prev2_open']
    second_small_body = abs(df['prev1_close'] - df['prev1_open']) < abs(df['prev2_close'] - df['prev2_open']) * 0.5
    small_body_gapped_down = (df['prev1_low'] < df['prev2_low']) & (df['prev1_high'] < df['prev2_low'])
    third_bullish = df[close_col] > df[open_col]
    third_closes_above_midpoint = df[close_col] > ((df['prev2_open'] + df['prev2_close']) / 2)
    
    df[morning_star_label] = (first_bearish & second_small_body & small_body_gapped_down & third_bullish & third_closes_above_midpoint).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_morning_star(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLONNECK = """
import pandas as pd

def calculate_on_neck(df, open_col='open_price', close_col='close_price', low_col='low_price'):
    on_neck_label = 'CDLONNECK'
    
    # Shift the open, close, and low prices to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    # On-Neck Pattern:
    # - The previous candle is a long bearish candle.
    # - The current candle opens below the previous candle's low.
    # - The current candle closes at or near the previous candle's low.
    previous_long_bearish = df['prev_close'] < df['prev_open']
    current_opens_below_prev_low = df[open_col] < df['prev_low']
    current_closes_at_or_near_prev_low = (df[close_col] <= df['prev_low'] * 1.001) & (df[close_col] >= df['prev_low'] * 0.999)
    
    df[on_neck_label] = (previous_long_bearish & current_opens_below_prev_low & current_closes_at_or_near_prev_low).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
low_col = 'low_price'
calculate_on_neck(dataset, open_col=open_col, close_col=close_col, low_col=low_col)
"""

CDLRICKSHAWMAN = """
import pandas as pd

def calculate_rickshaw_man(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    rickshaw_man_label = 'CDLRICKSHAWMAN'
    
    # Calculate the real body size
    real_body = abs(df[close_col] - df[open_col])
    
    # Rickshaw Man:
    # - Very small real body
    # - Long upper and lower shadows
    # - Open and close prices are near the middle of the trading range
    df[rickshaw_man_label] = (
        (real_body <= (df[high_col] - df[low_col]) * threshold) &
        (abs(df[high_col] - df[open_col]) > real_body * 2) &
        (abs(df[low_col] - df[open_col]) > real_body * 2) &
        (abs(df[close_col] - df[open_col]) <= (df[high_col] - df[low_col]) * 0.1)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_rickshaw_man(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""

CDLRISEFALL3METHODS = """
import pandas as pd

def calculate_rising_falling_three_methods(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price']):
    rising_three_methods_label = 'CDLRISE3METHODS'
    falling_three_methods_label = 'CDLFALL3METHODS'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    df['prev3_open'] = df[open_col].shift(3)
    df['prev3_close'] = df[close_col].shift(3)
    df['prev3_high'] = df[high_col].shift(3)
    df['prev3_low'] = df[low_col].shift(3)
    
    df['prev4_open'] = df[open_col].shift(4)
    df['prev4_close'] = df[close_col].shift(4)
    df['prev4_high'] = df[high_col].shift(4)
    df['prev4_low'] = df[low_col].shift(4)
    
    # Rising Three Methods:
    # - The first candle is a long bullish candle.
    # - The next three candles are small bearish candles that remain within the range of the first candle.
    # - The fifth candle is a long bullish candle that closes above the first candle's close.
    first_long_bullish = df['prev4_close'] > df['prev4_open']
    three_small_bearish = (df['prev3_close'] < df['prev3_open']) & (df['prev2_close'] < df['prev2_open']) & (df['prev1_close'] < df['prev1_open'])
    within_first_candle_range = (df['prev3_low'] > df['prev4_low']) & (df['prev3_high'] < df['prev4_high']) & \
                                (df['prev2_low'] > df['prev4_low']) & (df['prev2_high'] < df['prev4_high']) & \
                                (df['prev1_low'] > df['prev4_low']) & (df['prev1_high'] < df['prev4_high'])
    fifth_long_bullish = df[close_col] > df[open_col]
    fifth_closes_above_first_close = df[close_col] > df['prev4_close']
    
    df[rising_three_methods_label] = (first_long_bullish & three_small_bearish & within_first_candle_range & fifth_long_bullish & fifth_closes_above_first_close).astype(int)
    
    # Falling Three Methods:
    # - The first candle is a long bearish candle.
    # - The next three candles are small bullish candles that remain within the range of the first candle.
    # - The fifth candle is a long bearish candle that closes below the first candle's close.
    first_long_bearish = df['prev4_close'] < df['prev4_open']
    three_small_bullish = (df['prev3_close'] > df['prev3_open']) & (df['prev2_close'] > df['prev2_open']) & (df['prev1_close'] > df['prev1_open'])
    within_first_candle_range = (df['prev3_low'] > df['prev4_low']) & (df['prev3_high'] < df['prev4_high']) & \
                                (df['prev2_low'] > df['prev4_low']) & (df['prev2_high'] < df['prev4_high']) & \
                                (df['prev1_low'] > df['prev4_low']) & (df['prev1_high'] < df['prev4_high'])
    fifth_long_bearish = df[close_col] < df[open_col]
    fifth_closes_below_first_close = df[close_col] < df['prev4_close']
    
    df[falling_three_methods_label] = (first_long_bearish & three_small_bullish & within_first_candle_range & fifth_long_bearish & fifth_closes_below_first_close).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low', 
                     'prev3_open', 'prev3_close', 'prev3_high', 'prev3_low', 
                     'prev4_open', 'prev4_close', 'prev4_high', 'prev4_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_rising_falling_three_methods(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLSEPARATINGLINES = """
import pandas as pd

def calculate_separating_lines(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price']):
    separating_lines_label = 'CDLSEPARATINGLINES'
    
    # Shift the necessary columns to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    
    # Bullish Separating Lines:
    # - The previous candle is a bearish candle.
    # - The current candle is a bullish candle that opens at the previous candle's open price.
    prev_bearish = df['prev_close'] < df['prev_open']
    curr_bullish = df[close_col] > df[open_col]
    bullish_separating_lines = prev_bearish & curr_bullish & (df[open_col] == df['prev_open'])
    
    # Bearish Separating Lines:
    # - The previous candle is a bullish candle.
    # - The current candle is a bearish candle that opens at the previous candle's open price.
    prev_bullish = df['prev_close'] > df['prev_open']
    curr_bearish = df[close_col] < df[open_col]
    bearish_separating_lines = prev_bullish & curr_bearish & (df[open_col] == df['prev_open'])
    
    df[separating_lines_label] = (bullish_separating_lines | bearish_separating_lines).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
calculate_separating_lines(dataset, open_col=open_col, close_col=close_col)
"""

CDLSHOOTINGSTAR = """
import pandas as pd

def calculate_shooting_star(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_threshold=0.001, shadow_ratio=2):
    shooting_star_label = 'CDLSHOOTINGSTAR'
    
    # Calculate the real body size and the upper shadow size
    real_body = abs(df[close_col] - df[open_col])
    upper_shadow = df[high_col] - df[close_col].where(df[close_col] > df[open_col], df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    
    # Shooting Star:
    # - The real body is small.
    # - The upper shadow is at least twice the size of the real body.
    # - The lower shadow is very small or non-existent.
    # - The real body is located at the lower end of the trading range.
    df[shooting_star_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) &
        (upper_shadow >= real_body * shadow_ratio) &
        (lower_shadow <= real_body)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_threshold = 0.001
shadow_ratio = 2
calculate_shooting_star(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLSHORTLINE = """
import pandas as pd

def calculate_short_line_candle(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_ratio=0.3):
    short_line_candle_label = 'CDLSHORTLINE'
    
    # Calculate the real body size and the range (high - low)
    real_body = abs(df[close_col] - df[open_col])
    candle_range = df[high_col] - df[low_col]
    
    # Short Line Candle:
    # - The real body is small relative to the range of the candle
    df[short_line_candle_label] = (real_body <= candle_range * body_ratio).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_ratio = 0.3
calculate_short_line_candle(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_ratio=body_ratio)
"""

CDLSPINNINGTOP = """
import pandas as pd

def calculate_spinning_top(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_ratio=0.3, shadow_ratio=0.1):
    spinning_top_label = 'CDLSPINNINGTOP'
    
    # Calculate the real body size
    real_body = abs(df[close_col] - df[open_col])
    
    # Calculate the upper and lower shadow sizes
    upper_shadow = df[high_col] - df[[close_col, open_col]].max(axis=1)
    lower_shadow = df[[close_col, open_col]].min(axis=1) - df[low_col]
    
    # Spinning Top:
    # - The real body is small relative to the total range of the candle
    # - The upper and lower shadows are larger compared to the real body
    df[spinning_top_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_ratio) &
        (upper_shadow >= real_body * shadow_ratio) &
        (lower_shadow >= real_body * shadow_ratio)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_ratio = 0.3
shadow_ratio = 0.1
calculate_spinning_top(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_ratio=body_ratio, shadow_ratio=shadow_ratio)
"""

CDLSTALLEDPATTERN = """
import pandas as pd

def calculate_stalled_pattern(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    stalled_pattern_label = 'CDLSTALLEDPATTERN'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    df['prev3_open'] = df[open_col].shift(3)
    df['prev3_close'] = df[close_col].shift(3)
    df['prev3_high'] = df[high_col].shift(3)
    df['prev3_low'] = df[low_col].shift(3)
    
    # Stalled Pattern (Deliberation):
    # - The first candle is a long bullish candle.
    # - The second candle is also bullish, with a higher close and open than the first candle.
    # - The third candle is a small bullish candle that opens within the body of the second candle and closes near the open of the second candle.
    first_long_bullish = df['prev3_close'] > df['prev3_open']
    second_bullish_higher = (df['prev2_close'] > df['prev2_open']) & (df['prev2_close'] > df['prev3_close']) & (df['prev2_open'] > df['prev3_open'])
    third_small_bullish = (df['prev1_close'] > df['prev1_open']) & (df['prev1_open'] < df['prev2_close']) & (df['prev1_close'] < df['prev2_close'])
    third_close_near_second_open = abs(df['prev1_close'] - df['prev2_open']) < (df['prev2_close'] - df['prev2_open']) * 0.5
    
    df[stalled_pattern_label] = (first_long_bullish & second_bullish_higher & third_small_bullish & third_close_near_second_open).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low', 
                     'prev3_open', 'prev3_close', 'prev3_high', 'prev3_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_stalled_pattern(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLSTICKSANDWICH = """
import pandas as pd

def calculate_stick_sandwich(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    stick_sandwich_label = 'CDLSTICKSANDWICH'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Stick Sandwich Pattern:
    # - The first candle is a bearish candle.
    # - The second candle is a bullish candle that closes higher than the first candle's close.
    # - The third candle is a bearish candle that closes at the same level as the first candle's close.
    first_bearish = df['prev2_close'] < df['prev2_open']
    second_bullish = df['prev1_close'] > df['prev1_open']
    third_bearish = df[close_col] < df[open_col]
    third_closes_same_as_first = df[close_col] == df['prev2_close']
    
    df[stick_sandwich_label] = (first_bearish & second_bullish & third_bearish & third_closes_same_as_first).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_stick_sandwich(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLTAKURI = """
import pandas as pd

def calculate_takuri(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', body_threshold=0.001, shadow_ratio=3):
    takuri_label = 'CDLTAKURI'
    
    # Calculate the real body size and the lower shadow size
    real_body = abs(df[close_col] - df[open_col])
    lower_shadow = df[open_col].where(df[open_col] < df[close_col], df[close_col]) - df[low_col]
    upper_shadow = df[high_col] - df[close_col].where(df[close_col] > df[open_col], df[open_col])
    
    # Takuri (Dragonfly Doji with very long lower shadow):
    # - The real body is very small
    # - The lower shadow is at least three times the length of the real body
    # - The upper shadow is very small or non-existent
    df[takuri_label] = (
        (real_body <= (df[high_col] - df[low_col]) * body_threshold) &
        (lower_shadow >= real_body * shadow_ratio) &
        (upper_shadow <= real_body)
    ).astype(int)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
body_threshold = 0.001
shadow_ratio = 3
calculate_takuri(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, body_threshold=body_threshold, shadow_ratio=shadow_ratio)
"""

CDLTASUKIGAP = """
import pandas as pd

def calculate_tasuki_gap(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    tasuki_gap_label = 'CDLTASUKIGAP'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Bullish Tasuki Gap:
    # - The first candle is a long bullish candle.
    # - The second candle is a bullish candle that opens above the high of the first candle, creating a gap.
    # - The third candle is a bearish candle that opens within the body of the second candle and closes within the gap created by the first and second candles but does not close the gap.
    bullish_first_long = df['prev2_close'] > df['prev2_open']
    bullish_second_gap = (df['prev1_open'] > df['prev2_high']) & (df['prev1_close'] > df['prev1_open'])
    bearish_third_in_gap = (df[close_col] < df[open_col]) & (df[open_col] > df['prev1_close']) & (df[close_col] < df['prev1_open']) & (df[close_col] > df['prev2_high'])
    
    bullish_tasuki_gap = bullish_first_long & bullish_second_gap & bearish_third_in_gap
    
    # Bearish Tasuki Gap:
    # - The first candle is a long bearish candle.
    # - The second candle is a bearish candle that opens below the low of the first candle, creating a gap.
    # - The third candle is a bullish candle that opens within the body of the second candle and closes within the gap created by the first and second candles but does not close the gap.
    bearish_first_long = df['prev2_close'] < df['prev2_open']
    bearish_second_gap = (df['prev1_open'] < df['prev2_low']) & (df['prev1_close'] < df['prev1_open'])
    bullish_third_in_gap = (df[close_col] > df[open_col]) & (df[open_col] < df['prev1_close']) & (df[close_col] > df['prev1_open']) & (df[close_col] < df['prev2_low'])
    
    bearish_tasuki_gap = bearish_first_long & bearish_second_gap & bullish_third_in_gap
    
    df[tasuki_gap_label] = (bullish_tasuki_gap | bearish_tasuki_gap).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_tasuki_gap(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLTHRUSTING = """
import pandas as pd

def calculate_thrusting_pattern(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    thrusting_pattern_label = 'CDLTHRUSTING'
    
    # Shift the necessary columns to compare with the previous day
    df['prev_open'] = df[open_col].shift(1)
    df['prev_close'] = df[close_col].shift(1)
    df['prev_high'] = df[high_col].shift(1)
    df['prev_low'] = df[low_col].shift(1)
    
    # Thrusting Pattern:
    # - The previous candle is a long bearish candle.
    # - The current candle opens below the previous candle's low.
    # - The current candle closes above the previous candle's midpoint but below the previous candle's open.
    previous_long_bearish = df['prev_close'] < df['prev_open']
    current_opens_below_prev_low = df[open_col] < df['prev_low']
    current_closes_within_range = (df[close_col] > (df['prev_open'] + df['prev_close']) / 2) & (df[close_col] < df['prev_open'])
    
    df[thrusting_pattern_label] = (previous_long_bearish & current_opens_below_prev_low & current_closes_within_range).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev_open', 'prev_close', 'prev_high', 'prev_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_thrusting_pattern(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLTRISTAR = """
import pandas as pd

def calculate_tristar_pattern(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price', threshold=0.001):
    tristar_pattern_label = 'CDLTRISTAR'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Define doji (where open and close prices are nearly equal)
    is_doji = lambda open_, close_: abs(open_ - close_) <= ((df[high_col] - df[low_col]) * threshold)
    
    # Tristar Pattern:
    # - The first, second, and third candles are Doji
    # - The second Doji gaps below or above the first and third Doji
    
    first_doji = is_doji(df['prev2_open'], df['prev2_close'])
    second_doji = is_doji(df['prev1_open'], df['prev1_close'])
    third_doji = is_doji(df[open_col], df[close_col])
    
    bullish_tristar = (first_doji & second_doji & third_doji &
                       (df['prev1_low'] > df['prev2_high']) &
                       (df[low_col] > df['prev1_high']))
    
    bearish_tristar = (first_doji & second_doji & third_doji &
                       (df['prev1_high'] < df['prev2_low']) &
                       (df[high_col] < df['prev1_low']))
    
    df[tristar_pattern_label] = (bullish_tristar | bearish_tristar).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
threshold = 0.001
calculate_tristar_pattern(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col, threshold=threshold)
"""

CDLUNIQUE3RIVER = """
import pandas as pd

def calculate_unique_3_river(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price'):
    unique_3_river_label = 'CDLUNIQUE3RIVER'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Unique 3 River Pattern:
    # - The first candle is a long bearish candle.
    # - The second candle is a small bullish or bearish candle with a lower low.
    # - The third candle is a small bullish candle that closes above the second candle's close but not higher than the first candle's open.
    first_long_bearish = df['prev2_close'] < df['prev2_open']
    second_small_candle = (df['prev1_open'] != df['prev1_close']) & (df['prev1_low'] < df['prev2_low'])
    third_small_bullish = (df[close_col] > df[open_col]) & (df[close_col] > df['prev1_close']) & (df[close_col] < df['prev2_open'])
    
    df[unique_3_river_label] = (first_long_bearish & second_small_candle & third_small_bullish).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_unique_3_river(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

CDLUPSIDEGAP2CROWS = """
import pandas as pd

def calculate_upside_gap_two_crows(df, open_col='open_price', close_col='close_price', high_col='high_price', low_col='low_price']):
    upside_gap_two_crows_label = 'CDLUPSIDEGAP2CROWS'
    
    # Shift the necessary columns to compare with the previous days
    df['prev1_open'] = df[open_col].shift(1)
    df['prev1_close'] = df[close_col].shift(1)
    df['prev1_high'] = df[high_col].shift(1)
    df['prev1_low'] = df[low_col].shift(1)
    
    df['prev2_open'] = df[open_col].shift(2)
    df['prev2_close'] = df[close_col].shift(2)
    df['prev2_high'] = df[high_col].shift(2)
    df['prev2_low'] = df[low_col].shift(2)
    
    # Upside Gap Two Crows Pattern:
    # - The first candle is a long bullish candle.
    # - The second candle is a bearish candle that opens above the high of the first candle, creating a gap, and closes within the body of the first candle.
    # - The third candle is a bearish candle that opens above the high of the second candle and closes below the close of the second candle but within the gap between the first and second candles.
    first_long_bullish = df['prev2_close'] > df['prev2_open']
    second_bearish_gaps_up = (df['prev1_open'] > df['prev2_high']) & (df['prev1_close'] < df['prev1_open']) & (df['prev1_close'] > df['prev2_close'])
    third_bearish_gaps_up = (df[open_col] > df['prev1_high']) & (df[close_col] < df['prev1_close']) & (df[close_col] > df['prev2_high'])
    
    df[upside_gap_two_crows_label] = (first_long_bullish & second_bearish_gaps_up & third_bearish_gaps_up).astype(int)
    
    # Drop the helper columns
    df.drop(columns=['prev1_open', 'prev1_close', 'prev1_high', 'prev1_low', 
                     'prev2_open', 'prev2_close', 'prev2_high', 'prev2_low'], inplace=True)

# Usage example:
open_col = 'open_price'
close_col = 'close_price'
high_col = 'high_price'
low_col = 'low_price'
calculate_upside_gap_two_crows(dataset, open_col=open_col, close_col=close_col, high_col=high_col, low_col=low_col)
"""

LOSING_CANDLES_ON_RANGE = """
import pandas as np

def calculate_losing_candles_indicator(df, column='close_price', lookback_range=[40, 10], threshold_percentage=50):
    def custom_agg(series):
        if len(series) < lookback_range[0]:
            return np.nan
        return series[-lookback_range[0]:-lookback_range[1]].sum()

    df['diff_helper'] = df[column].diff()
    df['is_decreasing_helper'] = (df['diff_helper'] < 0).astype(int)
    df['indicator_helper'] = df['is_decreasing_helper'].rolling(window=lookback_range[0], min_periods=lookback_range[0]).apply(custom_agg, raw=True)

    threshold_count = (threshold_percentage / 100) * (lookback_range[0] - lookback_range[1])
    range_start = lookback_range[0]
    range_end = lookback_range[1]
    df[f'losing_candles_{column}_from_{range_start}_to_{range_end}_threshold_{threshold_percentage}_perc'] = (df['indicator_helper'] >= threshold_count).astype(int)


    
    # Drop helper columns
    df.drop(columns=['diff_helper', 'is_decreasing_helper', 'indicator_helper'], inplace=True)

# Usage example:
lookback_range = [30, 1]
column = 'close_price'
threshold_percentage = 50
calculate_losing_candles_indicator(df=dataset, column=column, lookback_range=lookback_range, threshold_percentage=threshold_percentage)
"""

WINNING_CANDLES_ON_RANGE = """
import pandas as pd

def calculate_winning_candles_indicator(df, column='close_price', lookback_range=[40, 10], threshold_percentage=50):
    def custom_agg(series):
        if len(series) < lookback_range[0]:
            return np.nan
        return series[-lookback_range[0]:-lookback_range[1]].sum()

    df['diff_helper'] = df[column].diff()
    df['is_increasing_helper'] = (df['diff_helper'] > 0).astype(int)
    df['indicator_helper'] = df['is_increasing_helper'].rolling(window=lookback_range[0], min_periods=lookback_range[0]).apply(custom_agg, raw=True)

    threshold_count = (threshold_percentage / 100) * (lookback_range[0] - lookback_range[1])
    range_start = lookback_range[0]
    range_end = lookback_range[1]
    df[f'winning_candles_{column}_from_{range_start}_to_{range_end}_threshold_{threshold_percentage}_perc'] = (df['indicator_helper'] >= threshold_count).astype(int)

    # Drop helper columns
    df.drop(columns=['diff_helper', 'is_increasing_helper', 'indicator_helper'], inplace=True)

# Usage example:
lookback_range = [30, 1]
column = 'close_price'
threshold_percentage = 50
calculate_winning_candles_indicator(df=dataset, column=column, lookback_range=lookback_range, threshold_percentage=threshold_percentage)
"""

N_TH_PERCENTILE = """
import pandas as pd

def calculate_top_bottom_percent(df, column='close_price', lookback_period=20, percentile=10, position='top'):
    percentile_label = f"{column}_{percentile}th_{position}_{lookback_period}_lookback"

    if position == 'top':
        df[percentile_label] = (df[column].rolling(window=lookback_period)
                                          .apply(lambda x: x.iloc[-1] >= x.quantile(1 - percentile / 100.0), raw=False)
                                          .fillna(0)
                                          .astype(int))
    elif position == 'bottom':
        df[percentile_label] = (df[column].rolling(window=lookback_period)
                                          .apply(lambda x: x.iloc[-1] <= x.quantile(percentile / 100.0), raw=False)
                                          .fillna(0)
                                          .astype(int))
    else:
        raise ValueError("Position must be 'top' or 'bottom'")

# Usage example:
column = 'ROCP_1_close_price'
lookback_period = 500
percentile = 15
position = 'top'

calculate_top_bottom_percent(dataset, column=column, lookback_period=lookback_period, percentile=percentile, position=position)
"""

IS_NTH_DAY_OF_MONTH = """
import pandas as pd

def calculate_nth_day_of_month(df, date_col='kline_open_time', nth_day=5):
    # Convert the timestamp to datetime format
    df["temp_col"] = pd.to_datetime(df[date_col], unit='ms')
    
    # Extract the day of the month
    df['day_of_month'] = df["temp_col"].dt.day
    
    # Create a boolean indicator for the nth day of the month
    df[f'is_{nth_day}th_day_of_month'] = (df['day_of_month'] == nth_day).astype(int)
    
    # Drop the helper columns
    df.drop(['day_of_month', "temp_col"], axis=1, inplace=True)

# Usage example:
date_col = "kline_open_time"
nth_day = 5
calculate_nth_day_of_month(dataset, date_col=date_col, nth_day=nth_day)
"""


DEFAULT_CODE_PRESETS = [
    CodePreset(
        code=GEN_RSI_CODE,
        name="RSI",
        category=CodePresetCategories.INDICATOR,
        description="Generates RSI indicator on the selected column with the specified look-back periods.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=GEN_MA_CODE,
        name="SMA",
        category=CodePresetCategories.INDICATOR,
        description="Generates SMA indicator on the selected column with the specified look-back periods.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=GEN_TARGETS,
        name="TARGET",
        category=CodePresetCategories.INDICATOR,
        description="Generates forward looking target on the selected target column with the specified forward looking periods. This is useful for gauging correlations to future prices.",
        labels=CodePresetLabels.TARGET,
    ),
    CodePreset(
        code=GEN_ATR,
        name="ATR",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Average True Range (ATR) to measure market volatility over specified periods.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=IS_FRIDAY_8_UTC,
        name="IS_FRIDAY_8_UTC",
        category=CodePresetCategories.INDICATOR,
        description="Marks periods on Fridays between 8 and 10 UTC to help identify time-specific market behavior.",
        labels=CodePresetLabels.SEASONAL,
    ),
    CodePreset(
        code=GEN_OBV,
        name="OBV",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the On-Balance Volume (OBV) to track cumulative trading volume by adding or subtracting each period's volume based on the direction of the price movement.",
        labels=CodePresetLabels.VOLUME,
    ),
    CodePreset(
        code=GEN_HOURLY_FLAGS,
        name="HOURLY_FLAGS",
        category=CodePresetCategories.INDICATOR,
        description="Generates flags for each hour of the week, indicating if a particular record falls within that hour. Useful for time-series analysis.",
        labels=CodePresetLabels.SEASONAL,
    ),
    CodePreset(
        code=GEN_SIMPLE_CROSSING,
        name="SIMPLE_CROSSING",
        category=CodePresetCategories.INDICATOR,
        description="Detects when a value crosses above or below a specified threshold, marking the event for further analysis.",
        labels=CodePresetLabels.CROSSING,
    ),
    CodePreset(
        code=GEN_PERSISTENT_CROSSING,
        name="PERISTENT_CROSSING",
        category=CodePresetCategories.INDICATOR,
        description="Identifies persistent crossing events over a look-back period, enhancing reliability in trend identification.",
        labels=CodePresetLabels.CROSSING,
    ),
    CodePreset(
        code=GEN_BBANDS_CROSSING,
        name="BBANDS_CROSSING",
        category=CodePresetCategories.INDICATOR,
        description="Determines when prices cross above or below Bollinger Bands, which can signal significant market moves based on volatility and price levels.",
        labels=CodePresetLabels.OVERLAP,
    ),
    CodePreset(
        code=GEN_DEMA,
        name="DEMA",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Double Exponential Moving Average (DEMA) for a given period to provide a smoother and more responsive moving average.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=GEN_EMA,
        name="EMA",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Exponential Moving Average (EMA) which gives more weight to recent prices and reacts more quickly to price changes compared to a simple moving average. This is particularly useful for tracking trends in fast-moving markets.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=HT_TRENDLINE,
        name="HT_TRENDLINE",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Hilbert Transform - Instantaneous Trendline, which is a leading indicator used to generate signals reflecting the cyclical nature of price movements. This method is aimed to provide a smoothed version of the price data, focusing on the dominant market cycle component.",
        labels=CodePresetLabels.OVERLAP,
    ),
    CodePreset(
        code=KAMA,
        name="KAMA",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Kaufman Adaptive Moving Average (KAMA) which adapts to price volatility dynamically by using an efficiency ratio and smoothing constants to produce a more responsive moving average tailored to historical and current price movements.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=MAVP,
        name="MAVP",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Moving Average with Variable Periods (MAVP), allowing for different averaging periods at each data point, making it useful for datasets where the optimal averaging window changes over time.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=MIDPOINT,
        name="MIDPOINT",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the midpoint between the highest and lowest values of a given column over specified periods, representing a simple measure of the average of extreme values within the period.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=MIDPRICE,
        name="MIDPRICE",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Midpoint Price over a specified period. It averages the highest high and the lowest low for the given period, typically used to identify levels of support and resistance.",
        labels=CodePresetLabels.OVERLAP,
    ),
    CodePreset(
        code=PARABOLIC_SAR,
        name="PARABOLIC_SAR",
        category=CodePresetCategories.INDICATOR,
        description="The Parabolic SAR (Stop and Reverse) is used to determine potential reversals in the market price direction of traded assets like stocks or commodities. This indicator is particularly useful for setting trailing stops and identifying entry and exit points. It is plotted as a series of points above or below the price bars, signifying a potential reversal or continuation of the current trend.",
        labels=CodePresetLabels.OVERLAP,
    ),
    CodePreset(
        code=SAREXT,
        name="SAREXT",
        category=CodePresetCategories.INDICATOR,
        description="The Extended Parabolic SAR is an enhanced version of the standard Parabolic SAR. It is used to determine potential reversals in the market price direction of an asset, by accounting for volatility and acceleration. This version allows fine-tuning of the start, increment, and maximum acceleration factors, making it adaptable to different trading environments and volatility levels.",
        labels=CodePresetLabels.OVERLAP,
    ),
    CodePreset(
        code=T3_INDICATOR,
        name="T3",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Triple Exponential Moving Average (T3), which provides a smoother and more responsive moving average by applying multiple levels of exponential smoothing. This indicator is particularly useful for identifying the trend direction and strength.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=TEMA_INDICATOR,
        name="TEMA",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Triple Exponential Moving Average (TEMA), which uses triple smoothing to minimize the lag created by using multiple Exponential Moving Averages (EMAs), providing a more responsive moving average ideal for shorter trading periods or volatile markets.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=TRIMA,
        name="TRIMA",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Triangular Moving Average (TRIMA) over a specified period. This moving average is centered and smoothed to reduce lag, providing a clearer indication of the underlying trend.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=WMA,
        name="WMA",
        category=CodePresetCategories.INDICATOR,
        description="WMA (Weighted Moving Average): This indicator calculates the Weighted Moving Average, which gives more importance to recent data points than earlier ones. It's commonly used to identify trends more quickly than a simple moving average can.",
        labels=CodePresetLabels.SMOOTHING,
    ),
    CodePreset(
        code=ADX,
        name="ADX",
        category=CodePresetCategories.INDICATOR,
        description="The Average Directional Movement Index (ADX) quantifies trend strength by calculating a moving average of the price range expansion over a given period. A rising ADX indicates a strong trend, while a falling ADX suggests a weakening trend. The ADX is non-directional; it registers trend strength whether the price is trending upwards or downwards.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ADXR,
        name="ADXR",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Average Directional Movement Index Rating (ADXR) which is an average of the current ADX and the ADX from half the selected period ago. This helps in smoothing the ADX values and provides a clearer indication of trend strength and direction over time.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=APO,
        name="APO",
        category=CodePresetCategories.INDICATOR,
        description="The Absolute Price Oscillator (APO) is based on the difference between two exponential moving averages (EMAs) of a security's price, typically a fast and a slow EMA. The APO is used to identify momentum or trend strength by measuring the divergence between these EMAs.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=AROON_CUSTOM,
        name="AROON_CUSTOM",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Aroon indicator, which measures the time between highs and the time between lows over a given period. The indicator consists of two lines: Aroon Up (which measures the time since the last high) and Aroon Down (which measures the time since the last low). Both are expressed as a percentage of the total period. This indicator is useful for identifying trend changes and strength.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=AROON_CONVENTIONAL,
        name="AROON_CONVENTIONAL",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Aroon indicator, which measures the time between highs and the time between lows over a given period. The indicator consists of two lines: Aroon Up (which measures the time since the last high) and Aroon Down (which measures the time since the last low). Both are expressed as a percentage of the total period. This indicator is useful for identifying trend changes and strength.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=AROONOSC,
        name="AROONOSC",
        category=CodePresetCategories.INDICATOR,
        description="Computes the Aroon Oscillator to determine the trend strength and the likelihood of trend reversal. The oscillator fluctuates between -100 and +100, where high positive values indicate a strong uptrend, and high negative values suggest a strong downtrend.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=BOP,
        name="BOP",
        category=CodePresetCategories.INDICATOR,
        description="BOP (Balance of Power): This indicator is used to assess the strength of buyers and sellers by determining whether the price is being driven by buyers (closing near the high) or by sellers (closing near the low). It's particularly useful for identifying price divergences and market sentiment within trading periods.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=CCI,
        name="CCI",
        category=CodePresetCategories.INDICATOR,
        description="Commodity Channel Index (CCI): Calculates the CCI to identify cyclical trends within a dataset. The CCI compares the current price to an average price level over a specific time period with the normal deviations from that average. It helps in identifying overbought or oversold levels in price.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=CMO,
        name="CMO",
        category=CodePresetCategories.INDICATOR,
        description="Chande Momentum Oscillator (CMO): Measures the momentum of a security by comparing the sum of its recent gains to the sum of its recent losses. It is used to identify overbought and oversold conditions and potential reversals. The CMO ranges from -100 to +100, indicating overbought conditions at high positive values and oversold conditions at low negative values.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=DMI,
        name="DMI",
        category=CodePresetCategories.INDICATOR,
        description="Directional Movement Index (DMI): This indicator is designed to identify the directionality of the price movement. It includes the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI) to help determine the direction of the trend and the Average Directional Index (ADX) which measures the strength of the trend regardless of its direction.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MACD,
        name="MACD",
        category=CodePresetCategories.INDICATOR,
        description="MACD: Calculates the Moving Average Convergence/Divergence which consists of the MACD line (difference between two exponential moving averages), the signal line (an exponential moving average of the MACD line), and the MACD histogram (the difference between MACD and its signal line), used to expose changes in strength, direction, momentum, and duration of a trend in a stock's price. This function creates separate DataFrame columns with detailed labels for each element of the MACD to clearly indicate the parameters used in their computation.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MACDFIX,
        name="MACDFIX",
        category=CodePresetCategories.INDICATOR,
        description="MACDFIX: This customized version of the Moving Average Convergence Divergence (MACD) indicator includes detailed labels that reflect the calculation periods (12-day EMA and 26-day EMA for the MACD, and a 9-day EMA for the signal line) as well as the specific column name used for the calculation (e.g., 'close_price'). This makes the indicator outputs easily identifiable, especially when working with multiple data columns or various configurations of MACD in the same analysis. This version is highly useful for comparing performance across different datasets or within a dataset that includes multiple types of financial data.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MFI,
        name="MFI",
        category=CodePresetCategories.INDICATOR,
        description="Money Flow Index (MFI): Calculates the Money Flow Index over specified periods, which combines both price and volume data to measure trading pressure. A rising MFI indicates increased buying pressure, while a falling MFI suggests increased selling pressure.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MINUS_DI,
        name="MINUS_DI",
        category=CodePresetCategories.INDICATOR,
        description="Minus Directional Indicator (MINUS_DI): The Minus Directional Indicator measures the strength of downward price movement or trend. It is part of the Average Directional Index (ADX) system and is calculated by dividing the sum of downward price movement over a specified period by the total true range over the same period, then multiplying by 100.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MINUS_DM,
        name="MINUS_DM",
        category=CodePresetCategories.INDICATOR,
        description="Minus Directional Movement (Minus DM): This indicator is part of the Directional Movement System and is used to measure the downward price movement between periods. It focuses on the difference between the lows of two consecutive bars when the current low is lower than the previous low, and only if this difference is greater than the difference between the two highs. This indicator is typically used to assess the strength of a downtrend.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=MOM,
        name="MOM",
        category=CodePresetCategories.INDICATOR,
        description="Momentum (MOM): This indicator measures the rate of rise or fall in stock prices. It compares the current price to the price n periods ago and is used to identify the speed or strength of a price movement.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=PLUS_DI,
        name="PLUS_DI",
        category=CodePresetCategories.INDICATOR,
        description="Plus Directional Indicator (PLUS_DI): This indicator forms part of the Average Directional Movement Index system (ADX) and measures the presence of upward price movement. It is computed by comparing the current high with the previous high and is typically used to confirm trend direction. Higher values indicate a stronger upward trend.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=PLUS_DM,
        name="PLUS_DM",
        category=CodePresetCategories.INDICATOR,
        description="Plus Directional Movement (PLUS_DM): Helps to quantify the increase in the high price compared to the previous high, indicating the presence of upward price momentum when it outpaces the decreases in the low price. This indicator is used especially in combination with the ADX to measure trend strength.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=PPO,
        name="PPO",
        category=CodePresetCategories.INDICATOR,
        description="PPO (Percentage Price Oscillator): Calculates the Percentage Price Oscillator, which is used to show the difference between two exponential moving averages as a percentage of the larger moving average. This indicator is useful for identifying potential momentum shifts and trend reversals.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ROC,
        name="ROC",
        category=CodePresetCategories.INDICATOR,
        description="ROC (Rate of Change): Calculates the percentage change between the current price and the price a certain number of periods ago. It reflects the velocity of price changes and can be used to identify the strength of a trend.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ROCP,
        name="ROCP",
        category=CodePresetCategories.INDICATOR,
        description="ROCP: Calculates the Rate of Change Percentage, which measures the percentage change in price from one period to the next over specified periods to help identify momentum or speed of price changes.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ROCR,
        name="ROCR",
        category=CodePresetCategories.INDICATOR,
        description="ROCR: Calculates the Rate of Change Ratio which is the ratio of the current price to the price from a specified number of periods ago. This indicator helps to understand the magnitude of price changes and can be used to detect trends.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ROCR100,
        name="ROCR100",
        category=CodePresetCategories.INDICATOR,
        description="ROCR100: Calculates the Rate of Change Ratio 100, which compares the current price to the price from a previous period, scaled to 100. This indicator is used to show the percentage change in price relative to the prior period, assisting in identifying momentum in price movements.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=STOCH,
        name="STOCH",
        category=CodePresetCategories.INDICATOR,
        description="Stochastic Oscillator (STOCH): Compares the current closing price with its price range over a given period. The calculation results in two lines, %K and %D, which can help identify potential reversal points in the market as the price moves within this range.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=STOCHF,
        name="STOCHF",
        category=CodePresetCategories.INDICATOR,
        description="Stochastic Fast (STOCHF): This indicator helps to identify oversold and overbought conditions. The %K line is calculated from the current close minus the lowest low divided by the highest high minus the lowest low over the period. The %D line is the simple moving average of the %K line, serving as a signal line to interpret buy or sell signals.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=STOCHRSI,
        name="STOCHRSI",
        category=CodePresetCategories.INDICATOR,
        description="Stochastic RSI: The Stochastic Relative Strength Index (Stochastic RSI) is an oscillator used to identify overbought and oversold conditions by measuring the level of the RSI relative to its high-low range over a set period of time. It provides more sensitivity than the traditional RSI, offering quicker signals for entering and exiting trades. The StochRSI_K line is the Stochastic RSI itself, while the StochRSI_D line is a moving average of the StochRSI_K, typically used to generate signals when the two lines cross.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=TRIX,
        name="TRIX",
        category=CodePresetCategories.INDICATOR,
        description="TRIX: This indicator calculates the 1-day Rate-Of-Change (ROC) of a Triple Smooth Exponential Moving Average (EMA) of the specified column over a given period. TRIX helps to filter out insignificant price movements and identify underlying trends.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=ULTOSC,
        name="ULTOSC",
        category=CodePresetCategories.INDICATOR,
        description="Ultimate Oscillator (ULTOSC): This indicator combines buying pressure with true range over multiple time frames (short, intermediate, and long) to produce a value that reflects price momentum. The oscillator is typically used to identify bullish and bearish divergences which might indicate potential reversal points.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=WILLR,
        name="WILLR",
        category=CodePresetCategories.INDICATOR,
        description="Williams' %R: Calculates Williams' %R to determine overbought and oversold levels. Values range from -100 to 0, where readings near -100 typically indicate oversold conditions, and readings near 0 suggest overbought conditions. This can help identify potential reversal points in the price of an asset.",
        labels=CodePresetLabels.MOMENTUM,
    ),
    CodePreset(
        code=AD_LINE,
        name="AD_LINE",
        category=CodePresetCategories.INDICATOR,
        description="Chaikin A/D Line: The Chaikin Accumulation/Distribution Line helps track the volume flow in and out of stocks. A rising A/D line suggests accumulation (buying), indicating strong demand, while a falling A/D line indicates distribution (selling) and thus weaker demand. This indicator is often used to confirm trends or warn of potential reversals.",
        labels=CodePresetLabels.VOLUME,
    ),
    CodePreset(
        code=ADOSC,
        name="ADOSC",
        category=CodePresetCategories.INDICATOR,
        description="Chaikin A/D Oscillator (ADOSC): Measures the momentum of the Accumulation/Distribution line by calculating the difference between a fast and slow exponential moving average of the line. This indicator helps identify major changes in market sentiment and can signal potential reversals.",
        labels=CodePresetLabels.VOLUME,
    ),
    CodePreset(
        code=HT_DCPERIOD,
        name="HT_DCPERIOD",
        category=CodePresetCategories.INDICATOR,
        description="HT_DCPERIOD: This indicator calculates the Dominant Cycle Period using the Hilbert Transform, aiming to identify cycles in the price data by transforming the real price into a complex plane to detect cyclical components.",
        labels=CodePresetLabels.CYCLE,
    ),
    CodePreset(
        code=HT_DCPHASE,
        name="HT_DCPHASE",
        category=CodePresetCategories.INDICATOR,
        description="HT_DCPHASE: The Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) calculates the phase of the cycle within market data. This can help identify turning points in the market cycle, offering insights into potential shifts in momentum or trend changes.",
        labels=CodePresetLabels.CYCLE,
    ),
    CodePreset(
        code=HT_PHASOR,
        name="HT_PHASOR",
        category=CodePresetCategories.INDICATOR,
        description="HT_PHASOR: This indicator calculates the Hilbert Transform Phasor Components of a given data series. It provides the instantaneous phase and amplitude, which are useful for identifying cycles and the amplitude modulation of price movements within the financial market.",
        labels=CodePresetLabels.CYCLE,
    ),
    CodePreset(
        code=HT_SINE,
        name="HT_SINE",
        category=CodePresetCategories.INDICATOR,
        description="HT_SINE (Hilbert Transform - SineWave): This indicator is part of the Hilbert Transform suite used to generate an in-phase and quadrature component of the price cycle, typically helping to identify cycles and turning points in the market with sine and lead sine waves.",
        labels=CodePresetLabels.CYCLE,
    ),
    CodePreset(
        code=HT_TRENDMODE,
        name="HT_TRENDMODE",
        category=CodePresetCategories.INDICATOR,
        description="HT_TRENDMODE: This indicator is calculated using the Hilbert Transform to identify the dominant cycle phase of the price action, helping to distinguish between trend and cycle modes. A positive value suggests a trend mode, while a zero or negative value could suggest a cyclical or non-trending mode.",
        labels=CodePresetLabels.CYCLE,
    ),
    CodePreset(
        code=AVGPRICE,
        name="AVGPRICE",
        category=CodePresetCategories.INDICATOR,
        description="AVGPRICE: Calculates the Average Price (AVGPRICE) indicator by taking the mean of the high, low, and close prices for each period, providing a simple reflection of a typical trading price within a given timeframe.",
        labels=CodePresetLabels.PRICE_TRANSFORM,
    ),
    CodePreset(
        code=MEDPRICE,
        name="MEDPRICE",
        category=CodePresetCategories.INDICATOR,
        description="MEDPRICE: Calculates the Median Price, which is the average of the high and low prices for each period. It provides a simple measure of the midpoint of a security's trading range for the day and can be used as a foundation for other technical indicators.",
        labels=CodePresetLabels.PRICE_TRANSFORM,
    ),
    CodePreset(
        code=TYPPRICE,
        name="TYPPRICE",
        category=CodePresetCategories.INDICATOR,
        description="TYPPRICE: Calculates the Typical Price for each period, which is the average of the high, low, and close prices. This indicator is often used as a simplified representation of the price action and can be utilized as a reference for other calculations, such as moving averages or pivot points.",
        labels=CodePresetLabels.PRICE_TRANSFORM,
    ),
    CodePreset(
        code=WCLPRICE,
        name="WCLPRICE",
        category=CodePresetCategories.INDICATOR,
        description="WCLPRICE: Calculates the Weighted Close Price by taking the average of the high, low, and twice the closing price. This indicator emphasizes the closing price more than the high and low prices during the period, typically used to confirm trends or reversals.",
        labels=CodePresetLabels.PRICE_TRANSFORM,
    ),
    CodePreset(
        code=NATR,
        name="NATR",
        category=CodePresetCategories.INDICATOR,
        description="NATR: Calculates the Normalized Average True Range (NATR) to measure market volatility relative to the price. It expresses the ATR as a percentage of the closing price, helping to compare volatility across different price levels.",
        labels=CodePresetLabels.VOLATILITY,
    ),
    CodePreset(
        code=TRANGE,
        name="TRANGE",
        category=CodePresetCategories.INDICATOR,
        description="TRange: Calculates the True Range to assess the volatility. True Range is the greatest of the following: current high minus current low, the absolute value of the current high minus the previous close, and the absolute value of the current low minus the previous close.",
        labels=CodePresetLabels.VOLATILITY,
    ),
    CodePreset(
        code=CDL2CROWS,
        name="CDL2CROWS",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the Two Crows candlestick pattern, a bearish reversal pattern that appears in an uptrend. It consists of three candles: a long bullish candle, followed by a gap-up open candle that closes within the body of the first but does not surpass its high, and a third candle that opens higher than the second but closes below its open, signaling a potential reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLENGULFING,
        name="Engulfing pattern",
        category=CodePresetCategories.INDICATOR,
        description="Engulfing Pattern: This indicator identifies potential reversal points in price trends. A bullish engulfing pattern occurs when a smaller red (down) candle is followed by a larger green (up) candle, engulfing the previous day's body. This suggests a potential bullish reversal. Conversely, a bearish engulfing pattern appears when a green candle is followed by a larger red candle, signaling a potential bearish reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CLIP_FROM_RANGE,
        name="CLIP_FROM_RANGE",
        category=CodePresetCategories.INDICATOR,
        description="The function adjust_column_values_based_on_range is designed to selectively modify values within a specified column of a pandas DataFrame based on a defined numerical range. When a value in the column falls within this specified range, inclusive of the lower and upper bounds, it is reset to 0. Values outside this range are left unchanged. This functionality is particularly useful for data cleaning or normalization processes where certain value ranges are considered as outliers or require nullification for analytical reasons",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=THRESHOLD_FLAG,
        name="THRESHOLD_FLAG",
        category=CodePresetCategories.INDICATOR,
        description="Makes a new column based on some existing one where all the values will be (0 or 1) based on whether the value in the existing column surpasses some threshold. It is useful in cases where we want to ignore most normal cases but don't want the model to learn too much from very extreme values. One example column where this could be useful is 'volume'.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=DIFF_TWO,
        name="MINUS_FUNC",
        category=CodePresetCategories.INDICATOR,
        description="Calculates the difference between two columns on each data point.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=ROCP_DIFF,
        name="PERC_CHANGE_FUNC",
        category=CodePresetCategories.INDICATOR,
        description="Calculate the percentage change between two columns, useful for financial data to see relative changes over time.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=SUM_FUNC,
        name="SUM_FUNC",
        category=CodePresetCategories.INDICATOR,
        description="Sum two columns to combine data into a single metric, useful for aggregating related information.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=PROD_FUNC,
        name="PROD_FUNC",
        category=CodePresetCategories.INDICATOR,
        description="Calculate the product of two columns, which could be useful for creating interaction features in machine learning models.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=DIFF_SQRD_FUNC,
        name="DIFF_SQRD_FUNC",
        category=CodePresetCategories.INDICATOR,
        description="Compute the squared difference between two columns to emphasize larger differences and penalize them more heavily, suitable for certain statistical analyses.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=ROW_WISE_MIN_MAX,
        name="ROW_WISE_MIN_MAX",
        category=CodePresetCategories.INDICATOR,
        description="Determine the minimum or maximum of two columns at each row, useful for bounding values or finding extremes in data.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=TRIM_TO_NONE,
        name="TRIM_TO_NONE",
        category=CodePresetCategories.INDICATOR,
        description="Removes the first up to N values of a column if the column gets skewed results when it doesn't have a sufficient look-back period available yet. Useful for cleaning up RSI for example.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=COLS_CROSSING,
        name="COLS_CROSSING",
        category=CodePresetCategories.INDICATOR,
        description="This function determines the point when two columns cross. It generats two new columns: one when column A crosses above column B and also when column A crosses below column B.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=SALARY_DAY,
        name="SALARY_DAY",
        category=CodePresetCategories.INDICATOR,
        description="A flag that says whether the day is 2nd last day of the month. The global payday is thought to be the last day of the month.",
        labels=CodePresetLabels.SEASONAL,
    ),
    CodePreset(
        code=IS_MONDAY,
        name="IS_MONDAY",
        category=CodePresetCategories.INDICATOR,
        description="A flag that says whether current day is monday.",
        labels=CodePresetLabels.SEASONAL,
    ),
    CodePreset(
        code=EVENT_PERC_CHANGE_FLAG,
        name="PRICE_CHANGE_ON_EVENT_FLAG",
        category=CodePresetCategories.INDICATOR,
        description="Function that is provided a binary column (0 or 1) and computes whether significant enough ROCP happened on cases where the binary column's value is 1.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=INCREASING_STREAK,
        name="INCREASING_STREAK",
        category=CodePresetCategories.INDICATOR,
        description="The indicator sets the value to 1 if the specified column has been increasing for the previous N rows consecutively; otherwise, it sets the value to 0. This helps identify periods of consistent upward movement in the given column.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=DECREASING_STREAK,
        name="DECREASING_STREAK",
        category=CodePresetCategories.INDICATOR,
        description="The indicator sets the value to 1 if the specified column has been decreasing for the previous N rows consecutively; otherwise, it sets the value to 0. This helps identify periods of consistent downward movement in the given column.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=CONDITIONAL_FLAG,
        name="CONDITIONAL_FLAG",
        category=CodePresetCategories.INDICATOR,
        description="This indicator generates a flag indicating a specific market condition based on a customizable condition.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=ALL_CONDS_TRUE,
        name="ALL_CONDS_TRUE",
        category=CodePresetCategories.INDICATOR,
        description="This code generates an indicator column that is 1 if all specified condition columns are 1, and 0 otherwise.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=CONTAINS_TRUE_N_LOOKBACK,
        name="CONTAINS_TRUE_N_LOOKBACK",
        category=CodePresetCategories.INDICATOR,
        description="Indicator Description: This indicator creates a new column that flags a value of 1 if any of the previous n rows (including the current row) contain a 1 in the specified source column. This is useful for detecting recent occurrences of a specific condition over a defined look-back period.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=CDL3BLACKCROWS,
        name="CDL3BLACKCROWS",
        category=CodePresetCategories.INDICATOR,
        description="Three Black Crows (CDL3BLACKCROWS): This pattern identifies a potential bearish reversal in the market. It consists of three consecutive long-bodied candlesticks that have closed progressively lower with each session. Each candle opens within the body of the previous one and closes lower, indicating strong selling pressure.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDL3INSIDE,
        name="CDL3INSIDE",
        category=CodePresetCategories.INDICATOR,
        description="CDL3INSIDE: Identifies the Three Inside Up/Down candlestick pattern. This pattern consists of three consecutive candlesticks and is used to signal potential reversals in the market. A bullish pattern occurs after a downtrend, indicating a potential upward reversal, while a bearish pattern occurs after an uptrend, indicating a potential downward reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDL3LINESTRIKE,
        name="CDL3LINESTRIKE",
        category=CodePresetCategories.INDICATOR,
        description="Three-Line Strike (CDL3LINESTRIKE): Identifies the bullish or bearish Three-Line Strike pattern, which is a four-candle continuation pattern. The first three candles are in the direction of the trend, followed by a fourth candle that reverses the trend direction, but does not invalidate the prior trend.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDL3OUTSIDE,
        name="CDL3OUTSIDE",
        category=CodePresetCategories.INDICATOR,
        description="CDL3OUTSIDE: This indicator identifies the Three Outside Up/Down candlestick pattern. The Three Outside Up pattern is a bullish reversal pattern, indicating that a downtrend may be reversing to an uptrend. Conversely, the Three Outside Down pattern is a bearish reversal pattern, indicating that an uptrend may be reversing to a downtrend.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDL3STARINSOUTH,
        name="CDL3STARINSOUTH",
        category=CodePresetCategories.INDICATOR,
        description="Three Stars in the South: This is a bullish reversal candlestick pattern that consists of three consecutive small-bodied candles with progressively lower highs and lows, often indicating a potential market bottom and a forthcoming upward reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDL3WHITESOLDIERS,
        name="CDL3WHITESOLDIERS",
        category=CodePresetCategories.INDICATOR,
        description="CDL3WHITESOLDIERS: Identifies the 'Three Advancing White Soldiers' pattern, which is a bullish reversal pattern consisting of three consecutive long-bodied candlesticks that open within the previous candle's real body and close progressively higher",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLABANDONEDBABY,
        name="CDLABANDONEDBABY",
        category=CodePresetCategories.INDICATOR,
        description="Abandoned Baby: This pattern can be both bullish and bearish. It typically occurs at the end of a trend and signals a reversal. A bullish abandoned baby pattern consists of a downtrend where a bearish candlestick is followed by a doji (a candlestick with little to no body), which is then followed by a bullish candlestick with a gap in between the doji and the following candlestick. The bearish abandoned baby is the reverse, appearing at the end of an uptrend.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLADVANCEDBLOCK,
        name="CDLADVANCEDBLOCK",
        category=CodePresetCategories.INDICATOR,
        description="Advance Block: The Advance Block pattern is a bearish reversal pattern consisting of three consecutive bullish candles with each successive candle showing signs of weakening. The candles open within the previous candle's body and have progressively smaller real bodies, indicating a loss of bullish momentum and a potential upcoming bearish reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLBELTHOLD,
        name="CDLBELTHOLD",
        category=CodePresetCategories.INDICATOR,
        description="The Belt-hold pattern is a single candlestick pattern that signifies potential reversal. A Bullish Belt-hold appears when the opening price is equal to the lowest price of the day, and the close is significantly higher, indicating a possible upward reversal. Conversely, a Bearish Belt-hold appears when the opening price is equal to the highest price of the day, and the close is significantly lower, indicating a possible downward reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHIKKAKE,
        name="CDLHIKKAKE",
        category=CodePresetCategories.INDICATOR,
        description="The Hikkake Pattern is a price pattern used in technical analysis to identify potential reversals in the market. It involves an inside bar formation followed by a breakout in the opposite direction, which is then considered a false breakout, signaling a reversal. This pattern can indicate a potential change in the trend direction when identified.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLPIERCING,
        name="CDLPIERCING",
        category=CodePresetCategories.INDICATOR,
        description="The Piercing Pattern is a bullish candlestick pattern used in technical analysis to identify potential reversals in a downtrend. It occurs when a bearish candle is followed by a bullish candle that opens lower than the previous low but closes more than halfway up the body of the previous bearish candle. This pattern suggests that the selling pressure is decreasing and buying pressure is increasing, indicating a potential upward reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLBREAKAWAY,
        name="CDLBREAKAWAY",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Breakaway candlestick pattern, which consists of five candles. In a Bullish Breakaway, the first candle is bullish, followed by three bearish candles, and the fifth candle is bullish, closing above the open of the first candle. In a Bearish Breakaway, the pattern is reversed. This pattern can signal a potential reversal in the market trend.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLCLOSINGMARUBOZU,
        name="CDLCLOSINGMARUBOZU",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Closing Marubozu candlestick pattern, where the closing price is equal to the high price for a bullish pattern or equal to the low price for a bearish pattern. The bullish Closing Marubozu indicates strong buying pressure, while the bearish Closing Marubozu indicates strong selling pressure",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLCONCEALBABYSWALL,
        name="CDLCONCEALBABYSWALL",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Concealing Baby Swallow candlestick pattern, which typically consists of three bearish candles followed by a bullish candle that swallows the previous bearish candle. The pattern suggests a potential bullish reversal in the market trend, indicating that the downward pressure may be coming to an end.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLCOUNTERATTACK,
        name="CDLCOUNTERATTACK",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Counterattack candlestick pattern, which consists of two candles. In a Bullish Counterattack, the first candle is bearish, and the second candle is bullish with both candles having the same closing price. In a Bearish Counterattack, the pattern is reversed with the first candle being bullish and the second candle being bearish with both candles having the same closing price. This pattern suggests a potential reversal in the market trend",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLDARKCLOUDCOVER,
        name="CDLDARKCLOUDCOVER",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Dark Cloud Cover candlestick pattern, which consists of a bullish candle followed by a bearish candle. The bearish candle opens above the high of the previous bullish candle and closes below the midpoint of the previous candle's body but above its open. This pattern suggests a potential bearish reversal in the market trend, indicating that the upward momentum might be weakening",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=IS_BEAR_MARKET,
        name="IS_BEAR_MARKET",
        category=CodePresetCategories.INDICATOR,
        description="Bear Market Indicator: This indicator signals a bear market by marking 1 if the close price is below a specified coefficient (default 0.8) of an already calculated simple moving average (SMA) column. This helps to identify potential downward trends in the market.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=CDLDOJI,
        name="CDLDOJI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies Doji candlestick patterns, which occur when the open and close prices are very close to each other, typically within a small threshold of the total trading range (high - low) of the candle. A Doji candlestick can indicate market indecision and potential trend reversal",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLDOJISTAR,
        name="CDLDOJISTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies Doji Star candlestick patterns, which are characterized by a Doji that gaps below or above the previous candlestick. The Doji candlestick itself occurs when the open and close prices are very close to each other, within a small threshold of the total trading range (high - low) of the candle. A Doji Star can indicate potential trend reversals",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLDRAGONFLYDOJI,
        name="CDLDRAGONFLYDOJI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies Dragonfly Doji candlestick patterns, which occur when the open and close prices are very close to each other at the high of the candle. The Dragonfly Doji often suggests a potential reversal or indecision in the market, and is characterized by a long lower shadow and little to no upper shadow. The threshold parameter helps to define how close the open and close prices should be to be considered a Dragonfly Doji",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLENGULFING,
        name="CDLENGULFING",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies Engulfing candlestick patterns, which are significant reversal patterns. A bullish engulfing pattern forms when a small bearish candle is followed by a large bullish candle that completely engulfs the previous candle's body, suggesting a potential upward reversal. Conversely, a bearish engulfing pattern forms when a small bullish candle is followed by a large bearish candle that completely engulfs the previous candle's body, indicating a potential downward reversal. The indicator outputs 1 for bullish engulfing and -1 for bearish engulfing patterns",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLEVENINGDOJISTAR,
        name="CDLEVENINGDOJISTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Evening Doji Star candlestick pattern, a bearish reversal pattern that typically occurs at the top of an uptrend. It consists of three candles A long bullish candle. A Doji that gaps above the previous candle. A long bearish candle that closes below the midpoint of the first candle. The Doji represents market indecision, and the subsequent bearish candle confirms the reversal",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLEVENINGSTAR,
        name="CDLEVENINGSTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Evening Star candlestick pattern, a bearish reversal pattern that typically occurs at the top of an uptrend. It consists of three candles",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLGAPSIDESIDEWHITE,
        name="CDLGAPSIDESIDEWHITE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Up-gap and Down-gap side-by-side white lines pattern, which is a continuation pattern. It consists of two consecutive white (bullish) candlesticks that gap up or down from the previous white bullish candlestick. This pattern suggests a continuation of the current trend. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLGRAVESTONEDOJI,
        name="CDLGRAVESTONEDOJI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Gravestone Doji candlestick pattern, which is a bearish reversal pattern that typically occurs at the top of an uptrend. It is characterized by a long upper shadow and little to no real body and lower shadow. The open, high, and close prices are very close to each other and at the high of the candle. The threshold parameter helps to define how close these prices should be to be considered a Gravestone Doji",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHAMMER,
        name="CDLHAMMER",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Hammer candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It is characterized by a small real body at the upper end of the trading range, with a long lower shadow at least twice the length of the real body. The threshold parameter helps to define how small the real body should be relative to the trading range, and the shadow_ratio parameter defines the minimum length of the lower shadow relative to the real body.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHANGINGMAN,
        name="CDLHANGINGMAN",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Hanging Man candlestick pattern, which is a bearish reversal pattern typically found at the top of an uptrend. It is characterized by a small real body at the upper end of the trading range, with a long lower shadow at least twice the length of the real body. The threshold parameter helps to define how small the real body should be relative to the trading range, and the shadow_ratio parameter defines the minimum length of the lower shadow relative to the real body",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHARAMI,
        name="CDLHARAMI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Harami candlestick pattern, which is a potential reversal pattern. A Bullish Harami occurs when a small bullish candle forms within the body of the previous bearish candle, suggesting a potential upward reversal. Conversely, a Bearish Harami occurs when a small bearish candle forms within the body of the previous bullish candle, indicating a potential downward reversal. The indicator outputs 1 for bullish harami and -1 for bearish harami patterns",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHARAMICROSS,
        name="CDLHARAMICROSS",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Harami Cross candlestick pattern, which is a significant reversal pattern. A Bullish Harami Cross occurs when a small Doji candle forms within the body of the previous bearish candle, suggesting a potential upward reversal. Conversely, a Bearish Harami Cross occurs when a small Doji candle forms within the body of the previous bullish candle, indicating a potential downward reversal. The indicator outputs 1 for bullish harami cross and -1 for bearish harami cross patterns",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHIGHWAVE,
        name="CDLHIGHWAVE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the High-Wave candlestick pattern, which is characterized by a very small real body and long upper and lower shadows. This pattern indicates a high level of market indecision and is often found at market turning points. The body_threshold parameter defines how small the real body should be relative to the trading range, and the shadow_ratio parameter defines the minimum length of the shadows relative to the real body",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHIKKAKEMOD,
        name="CDLHIKKAKEMOD",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Modified Hikkake Pattern, which is a complex reversal pattern. It starts with an inside day, where the current day's high is lower than the previous day's high and the current day's low is higher than the previous day's low. The pattern is confirmed if, on the third day, the price breaks out of the range of the inside day. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLHOMINGPIGEON,
        name="CDLHOMINGPIGEON",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Homing Pigeon candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It consists of two consecutive bearish candles, with the second candle's body entirely within the body of the first candle. This pattern suggests that the selling pressure is diminishing and a potential upward reversal may occur. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLIDENTICAL3CROWS,
        name="CDLIDENTICAL3CROWS",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Identical Three Crows candlestick pattern, which is a bearish reversal pattern. It consists of three consecutive long bearish candles where each candle opens within the real body of the previous candle and closes lower than the previous candle's close. This pattern suggests strong and consistent selling pressure, indicating a potential downward reversal. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLINNECK,
        name="CDLINNECK",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the In-Neck candlestick pattern, which is a bearish continuation pattern typically found in a downtrend. It consists of two candles: a long bearish candle followed by a smaller bullish or bearish candle that opens below the previous candle's low and closes near or slightly above the previous candle's close. This pattern suggests that the downtrend may continue. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLINVERTEDHAMMER,
        name="CDLINVERTEDHAMMER",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Inverted Hammer candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It is characterized by a small real body at the lower end of the trading range, with a long upper shadow at least twice the length of the real body and little to no lower shadow. The body_threshold parameter helps to define how small the real body should be relative to the trading range, and the shadow_ratio parameter defines the minimum length of the upper shadow relative to the real body",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLKICKING,
        name="CDLKICKING",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Kicking candlestick pattern, which is a strong reversal pattern. It consists of two marubozu candles ",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLKICKINGBYLENGTH,
        name="CDLKICKINGBYLENGTH",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Kicking candlestick pattern, which is a strong reversal pattern. It consists of two marubozu candles, The direction of the pattern is determined by the length of the longer marubozu. If the second marubozu is longer and bullish, it indicates a strong bullish reversal. If the second marubozu is longer and bearish, it indicates a strong bearish reversal. The indicator outputs 1 for a bullish reversal and -1 for a bearish reversal.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLLADDERBOTTOM,
        name="CDLLADDERBOTTOM",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Ladder Bottom candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It consists of five candles, 1. The first three candles are long bearish candles with consecutively lower closes. 2. he fourth candle is a small bearish or bullish candle with a lower low and a higher close. 3. The fifth candle is a long bullish candle that closes above the fourth candle's high. This pattern suggests that the selling pressure is diminishing, and a potential upward reversal may occur. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLLONGLEGGEDDOJI,
        name="CDLLONGLEGGEDDOJI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Long Legged Doji candlestick pattern, which is a type of Doji with long upper and lower shadows. It indicates a high level of market indecision as the price fluctuated significantly during the period but closed very close to the open price. The pattern suggests that neither buyers nor sellers were able to gain control, which can be a signal for a potential reversal or continuation of the trend",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLLONGLINE,
        name="CDLLONGLINE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Long Line Candle pattern, which is characterized by a long real body relative to the total range of the candle. This pattern indicates strong buying or selling pressure, suggesting that the current trend may continue. The body_ratio parameter defines how large the real body should be relative to the total range of the candle. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLMATCHINGLOW,
        name="CDLMATCHINGLOW",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Matching Low candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It consists of two consecutive candles with the same or very similar close prices. This pattern suggests that the support level is holding, indicating a potential upward reversal. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLMATHOLD,
        name="CDLMATHOLD",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Mat Hold candlestick pattern, which is a bullish continuation pattern. It consists of five candles",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLMORNINGDOJISTAR,
        name="CDLMORNINGDOJISTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Morning Doji Star candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It consists of three candles, This pattern suggests a potential upward reversal after a downtrend. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLMORNINGSTAR,
        name="CDLMORNINGSTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Morning Star candlestick pattern, which is a bullish reversal pattern typically found at the bottom of a downtrend. It consists of three candles, This pattern suggests a potential upward reversal after a downtrend. The indicator outputs 1 when the pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLONNECK,
        name="CDLONNECK",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the On-Neck candlestick pattern, which is a bearish continuation pattern. It consists of two candles, The first candle is a long bearish candle.The second candle opens below the previous candle's low and closes at or near the previous candle's low.This pattern suggests that the downtrend may continue. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLRICKSHAWMAN,
        name="CDLRICKSHAWMAN",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Rickshaw Man candlestick pattern, which is similar to the Long-Legged Doji. It is characterized by a very small real body with long upper and lower shadows, where the open and close prices are near the middle of the trading range. This pattern suggests a high level of market indecision and can signal a potential reversal. The threshold parameter helps to define how small the real body should be relative to the trading range.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLRISEFALL3METHODS,
        name="CDLRISEFALL3METHODS",
        category=CodePresetCategories.INDICATOR,
        description="These indicators identify the Rising Three Methods and Falling Three Methods candlestick patterns, which are bullish and bearish continuation patterns, respectively.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSEPARATINGLINES,
        name="CDLSEPARATINGLINES",
        category=CodePresetCategories.INDICATOR,
        description="Separating Lines (CDLSEPARATINGLINES): This indicator identifies the Separating Lines candlestick pattern, which can be a bullish or bearish continuation pattern. Bullish Separating Lines:The previous candle is a bearish candle.The current candle is a bullish candle that opens at the previous candle's open price. Bearish Separating Lines:The previous candle is a bullish candle.The current candle is a bearish candle that opens at the previous candle's open price.These patterns suggest a continuation of the current trend. The indicator outputs 1 when either the bullish or bearish pattern is detected",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSHOOTINGSTAR,
        name="CDLSHOOTINGSTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Shooting Star candlestick pattern, which is a bearish reversal pattern typically found at the top of an uptrend. It consists of a single candle with the following characteristics:A small real body located at the lower end of the trading range.A long upper shadow at least twice the length of the real body.A very small or non-existent lower shadow.This pattern suggests that the market opened near its low, then rallied significantly during the session, but closed near its opening price, indicating potential bearish reversal. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSHORTLINE,
        name="CDLSHORTLINE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Short Line Candle pattern, which is characterized by a small real body relative to the total range of the candle. This pattern indicates a period of consolidation or indecision in the market. The body_ratio parameter defines how small the real body should be relative to the total range of the candle. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSPINNINGTOP,
        name="CDLSPINNINGTOP",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Spinning Top candlestick pattern, which is characterized by a small real body and relatively long upper and lower shadows. This pattern indicates indecision in the market, where neither the bulls nor the bears have gained control. The body_ratio parameter defines how small the real body should be relative to the total range of the candle, and the shadow_ratio parameter defines the minimum length of the shadows relative to the real body. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSTALLEDPATTERN,
        name="CDLSTALLEDPATTERN",
        category=CodePresetCategories.INDICATOR,
        description="Stalled Pattern (CDLSTALLEDPATTERN): This indicator identifies the Stalled candlestick pattern, which is a bullish reversal pattern that typically appears in an uptrend. It consists of three candles: The first candle is a long bullish candle.The second candle is also bullish, with a higher close and open than the first candle.The third candle is a small bullish candle that opens within the body of the second candle and closes near the open of the second candle.This pattern suggests that the uptrend may be losing momentum, indicating a potential reversal. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLSTICKSANDWICH,
        name="CDLSTICKSANDWICH",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Stick Sandwich candlestick pattern, which is a bullish reversal pattern. It consists of three candles:The first candle is a bearish candle.The second candle is a bullish candle that closes higher than the first candle's close.The third candle is a bearish candle that closes at the same level as the first candle's close.This pattern suggests that the selling pressure is diminishing, and a potential upward reversal may occur. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLTAKURI,
        name="CDLTAKURI",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Takuri candlestick pattern, which is similar to the Dragonfly Doji but with a very long lower shadow. It is characterized by:A very small real body.A lower shadow at least three times the length of the real body.A very small or non-existent upper shadow.This pattern suggests a potential bullish reversal, indicating that the price opened, dropped significantly, but then recovered to close near the opening price, showing strong buying pressure. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLTASUKIGAP,
        name="CDLTASUKIGAP",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Tasuki Gap candlestick pattern, which can be either bullish or bearish and is a continuation pattern.Bullish Tasuki Gap:The first candle is a long bullish candle.The second candle is a bullish candle that opens above the high of the first candle, creating a gap.The third candle is a bearish candle that opens within the body of the second candle and closes within the gap created by the first and second candles but does not close the gap.Bearish Tasuki Gap:The first candle is a long bearish candle.The second candle is a bearish candle that opens below the low of the first candle, creating a gap.The third candle is a bullish candle that opens within the body of the second candle and closes within the gap created by the first and second candles but does not close the gap.This pattern suggests a continuation of the current trend.The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLTHRUSTING,
        name="CDLTHRUSTING",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Thrusting candlestick pattern, which is a bearish continuation pattern. It consists of two candles:The first candle is a long bearish candle.The second candle opens below the previous candle's low and closes above the previous candle's midpoint but below the previous candle's open.This pattern suggests that the downtrend may continue. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLTRISTAR,
        name="CDLTRISTAR",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Tristar candlestick pattern, which can be a bullish or bearish reversal pattern. It consists of three consecutive Doji candles with the following characteristics:The first, second, and third candles are Doji (where the open and close prices are nearly equal).The second Doji gaps below (for bullish) or above (for bearish) the first and third Doji.This pattern suggests a potential reversal in the trend. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLUNIQUE3RIVER,
        name="CDLUNIQUE3RIVER",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Unique 3 River candlestick pattern, which is a bullish reversal pattern. It consists of three candles:The first candle is a long bearish candle.The second candle is a small bullish or bearish candle with a lower low.The third candle is a small bullish candle that closes above the second candle's close but not higher than the first candle's open.This pattern suggests a potential upward reversal after a downtrend. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=CDLUPSIDEGAP2CROWS,
        name="CDLUPSIDEGAP2CROWS",
        category=CodePresetCategories.INDICATOR,
        description="This indicator identifies the Upside Gap Two Crows candlestick pattern, which is a bearish reversal pattern. It consists of three candles:The first candle is a long bullish candle.The second candle is a bearish candle that opens above the high of the first candle, creating a gap, and closes within the body of the first candle.The third candle is a bearish candle that opens above the high of the second candle and closes below the close of the second candle but within the gap between the first and second candles.This pattern suggests a potential downward reversal after an uptrend. The indicator outputs 1 when the pattern is detected.",
        labels=CodePresetLabels.CANDLE_PATTERN,
    ),
    CodePreset(
        code=LOSING_CANDLES_ON_RANGE,
        name="LOSING_CANDLES_ON_RANGE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator is provided with a look-back window and threshold. If the look-back window has at least reached the threshold amount of losing candles, the indicator will be 1. Otherwise, the indicator will be 0.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=WINNING_CANDLES_ON_RANGE,
        name="WINNING_CANDLES_ON_RANGE",
        category=CodePresetCategories.INDICATOR,
        description="This indicator is provided with a look-back window and threshold. If the look-back window has at least reached the threshold amount of winning candles, the indicator will be 1. Otherwise, the indicator will be 0.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=N_TH_PERCENTILE,
        name="N_TH_PERCENTILE",
        category=CodePresetCategories.INDICATOR,
        description="Top/Bottom Percentile Indicator: This function calculates whether the current value of a specified column is within the top or bottom percentile of values over a given lookback period. This can be useful for identifying periods when the price is at a relatively extreme level compared to its recent history.",
        labels=CodePresetLabels.CUSTOM,
    ),
    CodePreset(
        code=IS_NTH_DAY_OF_MONTH,
        name="IS_NTH_DAY_OF_MONTH",
        category=CodePresetCategories.INDICATOR,
        description="Binary indicator that states whether the current day is the nth day of the month.",
        labels=CodePresetLabels.SEASONAL,
    ),
]


def generate_default_code_presets():
    CodePresetQuery.create_many([item.to_dict() for item in DEFAULT_CODE_PRESETS])
