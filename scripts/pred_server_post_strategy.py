import sys

PATH_TO_TESTS_MAIN_DIR = "tests/prediction_server"
PATH_TO_PRED_SERVER = "realtime/prediction_server/src"

sys.path.append(PATH_TO_TESTS_MAIN_DIR)
sys.path.append(PATH_TO_PRED_SERVER)

from fixtures.strategy import strategy_simple_1, create_short_strategy_simple_1
from orm import Session
from schema.strategy import StrategyQuery

simple_strat = strategy_simple_1()
simple_short_strat = create_short_strategy_simple_1()


strategies = [
    StrategyQuery.create_entry(simple_strat),
    StrategyQuery.create_entry(simple_short_strat),
]


print(f"Created strategies {strategies}")
