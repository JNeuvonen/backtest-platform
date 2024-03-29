from conf import TEST_RUN_PORT
from import_helper import pred_server_routers, strategy_router


MINUTE_IN_MS = 1000 * 60
HOUR_IN_MS = MINUTE_IN_MS * 60
ONE_DAY_IN_MS = HOUR_IN_MS * 24

Routers = pred_server_routers()
StrategyRouter = strategy_router()


class URL:
    BASE_URL = f"http://localhost:{TEST_RUN_PORT}"

    @classmethod
    def _strategy_route(cls):
        return cls.BASE_URL + Routers.V1_STRATEGY

    @classmethod
    def create_strategy(cls):
        return cls._strategy_route() + StrategyRouter.STRATEGY
