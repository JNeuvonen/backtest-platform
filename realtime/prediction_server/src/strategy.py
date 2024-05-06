from typing import List
from code_gen_templates import CodeTemplates, PyCode
from log import LogExceptionContext
from binance_utils import fetch_binance_klines
from constants import KLINES_MAX_TIME_RANGE
from schema.data_transformation import DataTransformation, DataTransformationQuery
from utils import replace_placeholders_on_code_templ
from schema.strategy import Strategy
from datetime import datetime


def gen_data_transformations_code(data_transformations: List[DataTransformation]):
    sorted_transformations = sorted(data_transformations, key=lambda x: x.strategy_id)

    data_transformations_code = PyCode()
    data_transformations_code.append_line("def make_data_transformations(dataset):")
    data_transformations_code.add_indent()
    for item in sorted_transformations:
        data_transformations_code.add_block(item.transformation_code)

    data_transformations_code.append_line("return dataset")
    return data_transformations_code.get()


def get_trading_decisions(strategy: Strategy):
    with LogExceptionContext(re_raise=False):
        results_dict = {
            "fetched_data": None,
            "transformed_data": None,
            "should_enter_trade": None,
            "should_exit_trade": None,
        }

        binance_klines = fetch_binance_klines(
            strategy.symbol, strategy.candle_interval, KLINES_MAX_TIME_RANGE
        )
        results_dict["fetched_data"] = binance_klines

        data_transformations = DataTransformationQuery.get_transformations_by_strategy(
            strategy.id
        )

        data_transformations_helper = {
            "{DATA_TRANSFORMATIONS_FUNC}": gen_data_transformations_code(
                data_transformations
            )
        }

        exec(
            replace_placeholders_on_code_templ(
                CodeTemplates.DATA_TRANSFORMATIONS, data_transformations_helper
            ),
            globals(),
            results_dict,
        )

        trade_decision_helper = {
            "{ENTER_TRADE_FUNC}": strategy.enter_trade_code,
            "{EXIT_TRADE_FUNC}": strategy.exit_trade_code,
        }

        exec(
            replace_placeholders_on_code_templ(
                CodeTemplates.GEN_TRADE_DECISIONS, trade_decision_helper
            ),
            globals(),
            results_dict,
        )

        return {
            "should_enter_trade": results_dict["should_enter_trade"],
            "should_close_trade": results_dict["should_exit_trade"],
            "is_on_pred_serv_err": False,
        }


def format_pred_loop_log_msg(active_strats: int, strats_on_error: int, strategies_info):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"```Prediction service loop info - Timestamp (UTC): {current_time}```"
    log_msg += "\n------------"
    log_msg += f"\nPrediction loop completed.\nActive strategies: {active_strats}.\nStrategies in error state: {strats_on_error}"

    log_msg += "\n------------"
    log_msg += "\nStrategies state breakdown:"

    for item in strategies_info:
        log_msg += "\n------------"
        log_msg += f"\nStrategy: `{item['name']}`"
        if item["trading_decisions"] is not None:
            log_msg += f"\nShould Enter Trade: `{item['trading_decisions']['should_enter_trade']}`"
            log_msg += f"\nShould Close Trade: `{item['trading_decisions']['should_close_trade']}`"
        else:
            log_msg += "\nTrading decisions not available for the strategy."

        log_msg += f"\nIn Position: `{item['in_position']}`"

        if item["is_on_error"] is True:
            log_msg += f"\nIs on Error: {item['is_on_error']}"

    return log_msg
