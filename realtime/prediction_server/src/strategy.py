from code_gen_templates import CodeTemplates
from log import LogExceptionContext
from binance_utils import fetch_binance_klines
from constants import SOFTWARE_VERSION
from schema.data_transformation import DataTransformationQuery
from utils import (
    calculate_timestamp_for_kline_fetch,
    gen_data_transformations_code,
    get_current_timestamp_ms,
    replace_placeholders_on_code_templ,
)
from schema.strategy import Strategy, StrategyQuery
from datetime import datetime


def get_trading_decisions(strategy: Strategy):
    with LogExceptionContext(re_raise=False):
        results_dict = {
            "fetched_data": None,
            "transformed_data": None,
            "should_enter_trade": None,
            "should_exit_trade": None,
        }

        binance_klines = fetch_binance_klines(
            strategy.symbol,
            strategy.candle_interval,
            calculate_timestamp_for_kline_fetch(
                strategy.num_req_klines, strategy.kline_size_ms
            ),
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


def format_pred_loop_log_msg(
    current_state_dict, strategies_info, last_trade_loop_completed_timestamp_ms
):
    active_strats = current_state_dict["active_strategies"]
    strats_on_error = current_state_dict["strats_on_error"]
    in_position = current_state_dict["strats_in_pos"]
    strats_wanting_to_enter = current_state_dict["strats_wanting_to_enter"]
    strats_wanting_to_close = current_state_dict["strats_wanting_to_close"]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_timestamp_ms = get_current_timestamp_ms()
    log_msg = f"```Prediction service (v{SOFTWARE_VERSION}) loop info - Timestamp (UTC): {current_time} - Time to complete {(current_timestamp_ms - last_trade_loop_completed_timestamp_ms) / 1000} sec```"

    log_msg += "\n------------"
    log_msg += f"\nPrediction loop completed.\nActive strategies: {active_strats}.\nStrategies in error state: {strats_on_error}"

    log_msg += "\n------------"
    log_msg += "\nStrategies state breakdown:"
    log_msg += f"""\nNum in  position: `{in_position}`"""
    log_msg += f"""\nNum predicting enter: `{strats_wanting_to_enter}`"""
    log_msg += f"""\nNum predicting close: `{strats_wanting_to_close}`"""

    strategies_in_pos = []

    if in_position > 0:
        for item in strategies_info:
            if item["in_position"] is True:
                strat_name = "`" + item["name"] + "`"
                strategies_in_pos.append(strat_name)

    formatted_strategies = ", ".join(strategies_in_pos)

    log_msg += "\n------------"
    log_msg += f"\nStrategies currently in a position: {formatted_strategies}"

    return log_msg


def update_strategies_state_dict(
    strategy, trading_decisions, strats_dict, strategies_info
):
    if not strategy.is_disabled:
        strats_dict["active_strategies"] += 1

    if strategy.is_in_position is True:
        strats_dict["strats_in_pos"] += 1

    if trading_decisions is not None:
        strategies_info.append(
            {
                "name": strategy.name,
                "trading_decisions": trading_decisions,
                "in_position": strategy.is_in_position,
                "is_on_error": False,
            }
        )

        if trading_decisions["should_enter_trade"] is True:
            strats_dict["strats_wanting_to_enter"] += 1
        elif (
            trading_decisions["should_close_trade"] is False
            and trading_decisions["should_enter_trade"] is False
        ):
            strats_dict["strats_still"] += 1

        elif trading_decisions["should_close_trade"] is True:
            strats_dict["strats_wanting_to_close"] += 1

    else:
        strategies_info.append(
            {
                "name": strategy.name,
                "trading_decisions": None,
                "in_position": strategy.is_in_position,
                "is_on_error": True,
            }
        )
        StrategyQuery.update_strategy(strategy.id, {"is_on_pred_serv_err": True})
        strats_dict["strats_on_error"] += 1
