class LogLevel:
    EXCEPTION = "exception"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


class LogSourceProgram:
    TRADING_CLIENT = 1
    PRED_SERVER = 2


class SlackWebhooks:
    OPS_LOG_BOT = "ops_log_bot"
    TRADE_LOG = "trade_bot"
