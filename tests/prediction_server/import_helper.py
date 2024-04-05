import sys

from conf import SERVICE_CODE_SOURCE_DIR


sys.path.append(SERVICE_CODE_SOURCE_DIR)
import main
import orm
from api.v1.strategy import RoutePaths as strat_router
from api.v1.log import RoutePaths as logs_router
from api.v1.account import RoutePaths as acc_router
from api.v1.trade import RoutePaths as trade_router_helper
from schema.api_key import APIKeyQuery


def start_server():
    main.start_server()


def stop_server():
    main.stop_server()


def drop_tables():
    engine = get_db_engine()
    orm.drop_tables(engine)


def strategy_router():
    return strat_router


def cloudlogs_router():
    return logs_router


def account_router():
    return acc_router


def trade_router():
    return trade_router_helper


def pred_server_routers():
    return main.Routers


def get_db_engine():
    return orm.engine


def db_delete_all_data():
    orm.db_delete_all_data()


def create_api_key_entry(api_key: str):
    APIKeyQuery.create_entry({"key": api_key})
