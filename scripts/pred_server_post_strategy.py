import sys


PATH_TO_TESTS_MAIN_DIR = "tests/prediction_server"
PATH_TO_PRED_SERVER = "realtime/prediction_server/src"

sys.path.append(PATH_TO_TESTS_MAIN_DIR)
sys.path.append(PATH_TO_PRED_SERVER)

from fixtures.strategy import (
    strategy_simple_1,
    create_short_strategy_simple_1,
    create_short_strategy_simple_2,
    create_strategy_with_syntax_err,
)
from orm import Session
from schema.strategy import StrategyQuery

simple_strat = strategy_simple_1()
simple_short_strat = create_short_strategy_simple_1()
simple_short_strat_2 = create_short_strategy_simple_2()
strat_with_syntax_error = create_strategy_with_syntax_err()


strategies = [
    StrategyQuery.create_entry(simple_strat),
    StrategyQuery.create_entry(simple_short_strat),
    StrategyQuery.create_entry(simple_short_strat_2),
    StrategyQuery.create_entry(strat_with_syntax_error),
]


print(f"Created strategies {strategies}")
