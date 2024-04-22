from query_code_preset import CodePresetQuery


class CodePresetCategories:
    INDICATOR = "backtest_create_columns"


class CodePreset:
    code: str
    name: str
    category: str

    def __init__(self, code, name, category) -> None:
        self.code = code
        self.name = name
        self.category = category

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name, "category": self.category}


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


DEFAULT_CODE_PRESETS = [
    CodePreset(code=GEN_RSI_CODE, name="RSI", category=CodePresetCategories.INDICATOR),
    CodePreset(code=GEN_MA_CODE, name="SMA", category=CodePresetCategories.INDICATOR),
    CodePreset(
        code=GEN_TARGETS, name="TARGET", category=CodePresetCategories.INDICATOR
    ),
    CodePreset(code=GEN_ATR, name="ATR", category=CodePresetCategories.INDICATOR),
    CodePreset(
        code=IS_FRIDAY_8_UTC,
        name="IS_FRIDAY_8_UTC",
        category=CodePresetCategories.INDICATOR,
    ),
    CodePreset(
        code=GEN_OBV,
        name="OBV",
        category=CodePresetCategories.INDICATOR,
    ),
    CodePreset(
        code=GEN_HOURLY_FLAGS,
        name="HOURLY_FLAGS",
        category=CodePresetCategories.INDICATOR,
    ),
    CodePreset(
        code=GEN_SIMPLE_CROSSING,
        name="SIMPLE_CROSSING",
        category=CodePresetCategories.INDICATOR,
    ),
    CodePreset(
        code=GEN_PERSISTENT_CROSSING,
        name="PERISTENT_CROSSING",
        category=CodePresetCategories.INDICATOR,
    ),
    CodePreset(
        code=GEN_BBANDS_CROSSING,
        name="BBANDS_CROSSING",
        category=CodePresetCategories.INDICATOR,
    ),
]


def generate_default_code_presets():
    CodePresetQuery.create_many([item.to_dict() for item in DEFAULT_CODE_PRESETS])
