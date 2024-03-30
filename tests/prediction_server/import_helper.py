import sys

from conf import SERVICE_CODE_SOURCE_DIR


sys.path.append(SERVICE_CODE_SOURCE_DIR)
import main
import orm
from api.v1.strategy import RoutePaths as strat_router


def start_server():
    main.start_server()


def stop_server():
    main.stop_server()


def drop_tables():
    engine = get_db_engine()
    orm.drop_tables(engine)


def strategy_router():
    return strat_router


def pred_server_routers():
    return main.Routers


def get_db_engine():
    return orm.engine


def db_delete_all_data():
    orm.db_delete_all_data()
