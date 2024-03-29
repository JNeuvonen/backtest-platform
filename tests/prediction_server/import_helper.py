import sys

from conf import SERVICE_CODE_SOURCE_DIR


sys.path.append(SERVICE_CODE_SOURCE_DIR)
import main
import orm
from api.v1.strategy import RoutePaths as strat_router


def start_server():
    main.start_server()


def drop_tables():
    orm.drop_tables()


def strategy_router():
    return strat_router


def pred_server_routers():
    return main.Routers
