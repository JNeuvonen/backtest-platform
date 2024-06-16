class LogLevel:
    EXCEPTION = "exception"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


class LogSourceProgram:
    TRADING_CLIENT = 1
    PRED_SERVER = 2
    ANALYTICS_SERVER = 3


class SlackWebhooks:
    ALL_LOG_BOT = "log_bot_all"

    PRED_SERV_EXCEPTIONS = "pred_serv_exceptions"
    PRED_SERV_ALL = "pred_serv_all"

    ANALYTICS_EXCEPTIONS = "analytics_serv_exceptions"
    ANALYTICS_ALL = "analytics_serv_all"

    TRADE_CLIENT_EXCEPTIONS = "trade_client_exceptions"
    TRADE_CLIENT_ALL = "trade_client_all"

    TRADE_NOTIFS = "trade_notifications"


KLINES_MAX_TIME_RANGE = "1 Jan, 2017"


class TradeDirection:
    LONG = "LONG"
    SHORT = "SHORT"


SECOND_IN_MS = 1000
MINUTE_IN_MS = 1000 * 60
HOUR_IN_MS = MINUTE_IN_MS * 60
ONE_DAY_IN_MS = HOUR_IN_MS * 24
DAY_IN_MS = HOUR_IN_MS * 24
YEAR_IN_MS = DAY_IN_MS * 365
