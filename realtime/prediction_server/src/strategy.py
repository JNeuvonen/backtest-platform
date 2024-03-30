from code_gen_templates import CodeTemplates
from utils import replace_placeholders_on_code_templ
from schema.strategy import Strategy


def get_trading_decisions(strategy: Strategy):
    results_dict = {
        "fetched_data": None,
        "transformed_data": None,
        "should_enter_trade": None,
        "should_exit_trade": None,
    }

    fetch_data_helper = {"{FETCH_DATASOURCES_FUNC}": strategy.fetch_datasources_code}

    exec(
        replace_placeholders_on_code_templ(
            CodeTemplates.FETCH_DATASOURCES, fetch_data_helper
        ),
        globals(),
        results_dict,
    )

    data_transformations_helper = {
        "{DATA_TRANSFORMATIONS_FUNC}": strategy.data_transformations_code
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
    }
