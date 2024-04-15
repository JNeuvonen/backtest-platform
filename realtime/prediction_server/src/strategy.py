from typing import List
from code_gen_templates import CodeTemplates, PyCode
from log import LogExceptionContext
from schema.data_transformation import DataTransformation, DataTransformationQuery
from utils import replace_placeholders_on_code_templ
from schema.strategy import Strategy


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

        fetch_data_helper = {
            "{FETCH_DATASOURCES_FUNC}": strategy.fetch_datasources_code
        }

        exec(
            replace_placeholders_on_code_templ(
                CodeTemplates.FETCH_DATASOURCES, fetch_data_helper
            ),
            globals(),
            results_dict,
        )

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
