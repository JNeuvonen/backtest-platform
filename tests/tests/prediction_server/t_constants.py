from conf import TEST_RUN_PORT
from import_helper import (
    account_router,
    longshort_router,
    pred_server_routers,
    strategy_router,
    cloudlogs_router,
    trade_router,
)


MINUTE_IN_MS = 1000 * 60
HOUR_IN_MS = MINUTE_IN_MS * 60
ONE_DAY_IN_MS = HOUR_IN_MS * 24

Routers = pred_server_routers()
StrategyRouter = strategy_router()
LogsRouter = cloudlogs_router()
AccRouter = account_router()
TradeRouter = trade_router()
LongshortRouter = longshort_router()


class URL:
    BASE_URL = f"http://localhost:{TEST_RUN_PORT}"

    @classmethod
    def _strategy_route(cls):
        return cls.BASE_URL + Routers.V1_STRATEGY

    @classmethod
    def _cloudlogs_route(cls):
        return cls.BASE_URL + Routers.V1_LOGS

    @classmethod
    def _account_route(cls):
        return cls.BASE_URL + Routers.V1_ACC

    @classmethod
    def _trade_route(cls):
        return cls.BASE_URL + Routers.V1_TRADE

    @classmethod
    def _longshort_route(cls):
        return cls.BASE_URL + Routers.V1_LONG_SHORT

    @classmethod
    def create_strategy(cls):
        return cls._strategy_route() + StrategyRouter.STRATEGY

    @classmethod
    def create_longshort_strategy(cls):
        return cls._longshort_route() + LongshortRouter.LONG_SHORT

    @classmethod
    def create_trade(cls):
        return cls._trade_route() + TradeRouter.TRADE

    @classmethod
    def create_longshort_trade(cls, id):
        return cls._longshort_route() + LongshortRouter.LONG_SHORT_PAIR_ENTER.format(
            pair_id=id
        )

    @classmethod
    def exit_longshort_trade(cls, id):
        return cls._longshort_route() + LongshortRouter.LONG_SHORT_PAIR_EXIT.format(
            pair_id=id
        )

    @classmethod
    def fetch_strategies(cls):
        return cls._strategy_route() + StrategyRouter.STRATEGY

    @classmethod
    def fetch_trades(cls):
        return cls._trade_route() + TradeRouter.TRADE

    @classmethod
    def create_log(cls):
        return cls._cloudlogs_route() + LogsRouter.LOG

    @classmethod
    def fetch_logs(cls):
        return cls._cloudlogs_route() + LogsRouter.LOG

    @classmethod
    def create_account(cls):
        return cls._account_route() + AccRouter.ACCOUNT

    @classmethod
    def fetch_accounts(cls):
        return cls._account_route() + AccRouter.ACCOUNT

    @classmethod
    def fetch_account_by_name(cls, name):
        return cls._account_route() + AccRouter.ACCOUNT_BY_NAME.format(name=name)

    @classmethod
    def update_trade_close(cls, strat_id):
        return cls._strategy_route() + StrategyRouter.UPDATE_TRADE_CLOSE.format(
            id=strat_id
        )

    @classmethod
    def update_strategy(cls):
        return cls._strategy_route() + StrategyRouter.STRATEGY
