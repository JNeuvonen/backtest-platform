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

    ANALYTICS_EXCEPTIONS = "analytics_exceptions"
    ANALYTICS_ALL = "analytics_all"

    TRADE_CLIENT_EXCEPTIONS = "trade_client_exceptions"
    TRADE_CLIENT_ALL = "trade_client_all"

    TRADE_NOTIFS = "trade_notifications"


KLINES_MAX_TIME_RANGE = "1 Jan, 2017"


class TradeDirection:
    LONG = "LONG"
    SHORT = "SHORT"
