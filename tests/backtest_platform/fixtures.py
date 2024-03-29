from typing import List
from tests.backtest_platform.t_constants import BinanceCols
from tests.backtest_platform.t_utils import create_backtest_body, create_train_job_body

from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR
from code_gen import PyCode


def linear_model_basic():
    helper = PyCode()
    helper.append_line("class Model(nn.Module):")
    helper.add_indent()
    helper.append_line("def __init__(self, n_input_params):")
    helper.add_indent()
    helper.append_line("super(Model, self).__init__()")
    helper.append_line("self.linear = nn.Linear(n_input_params, 1)")
    helper.reduce_indent()
    helper.append_line("def forward(self, x):")
    helper.add_indent()
    helper.append_line("return self.linear(x)")
    helper.reduce_indent()
    helper.reduce_indent()
    return helper.get()


def criterion_basic():
    helper = PyCode()
    helper.append_line("def get_criterion_and_optimizer(model):")
    helper.add_indent()
    helper.append_line("criterion = nn.MSELoss()")
    helper.append_line(
        "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)"
    )
    helper.append_line("return criterion, optimizer")

    return helper.get()


NUM_EPOCHS_DEFAULT = 10


def create_train_job_basic(num_epochs=NUM_EPOCHS_DEFAULT):
    enter_trade_criteria = PyCode()
    enter_trade_criteria.append_line("def get_enter_trade_criteria(prediction):")
    enter_trade_criteria.add_indent()
    enter_trade_criteria.append_line("return prediction > 1.01")

    exit_trade_criteria = PyCode()
    enter_trade_criteria.append_line("def get_exit_trade_criteria(prediction):")
    enter_trade_criteria.add_indent()
    enter_trade_criteria.append_line("return prediction < 0.99")

    body = create_train_job_body(
        num_epochs=num_epochs,
        save_model_after_every_epoch=True,
        backtest_on_val_set=True,
        enter_trade_criteria=enter_trade_criteria.get(),
        exit_trade_criteria=exit_trade_criteria.get(),
    )
    return body


def create_backtest(dataset_name: str):
    enter_trade_cond = PyCode()
    exit_trade_cond = PyCode()

    enter_trade_cond.append_line("def get_enter_trade_criteria(prediction):")
    enter_trade_cond.add_indent()
    enter_trade_cond.append_line("return prediction > 1.01")

    exit_trade_cond.append_line("def get_exit_trade_criteria(prediction):")
    exit_trade_cond.add_indent()
    exit_trade_cond.append_line("return prediction < 0.99")

    return create_backtest_body(
        price_column=BinanceCols.OPEN_PRICE,
        epoch_nr=3,
        exit_trade_cond=exit_trade_cond.get(),
        enter_trade_cond=enter_trade_cond.get(),
        dataset_name=dataset_name,
    )


def create_manual_backtest(
    dataset_id: int,
    use_short_selling: bool,
    open_long_trade_cond: str,
    open_short_trade_cond: str,
    close_long_trade_cond: str,
    close_short_trade_cond: str,
    use_time_based_close: bool,
    trading_fees_perc: float,
    slippage_perc: float,
    backtest_data_range: List[int],
    use_profit_based_close: bool,
    use_stop_loss_based_close: bool,
    short_fee_hourly: float,
    take_profit_threshold_perc: float,
    stop_loss_threshold_perc: float,
):
    return {
        "dataset_id": dataset_id,
        "use_short_selling": use_short_selling,
        "open_long_trade_cond": open_long_trade_cond,
        "close_long_trade_cond": close_long_trade_cond,
        "open_short_trade_cond": open_short_trade_cond,
        "close_short_trade_cond": close_short_trade_cond,
        "use_time_based_close": use_time_based_close,
        "trading_fees_perc": trading_fees_perc,
        "slippage_perc": slippage_perc,
        "backtest_data_range": backtest_data_range,
        "use_profit_based_close": use_profit_based_close,
        "use_stop_loss_based_close": use_stop_loss_based_close,
        "short_fee_hourly": short_fee_hourly,
        "take_profit_threshold_perc": take_profit_threshold_perc,
        "stop_loss_threshold_perc": stop_loss_threshold_perc,
    }


def create_code_preset_body(code: str, category: str, name: str):
    return {"code": code, "category": category, "name": name}


def open_long_trade_cond_basic():
    enter_trade_cond = PyCode()

    enter_trade_cond.append_line("def open_long_trade(tick):")
    enter_trade_cond.add_indent()
    enter_trade_cond.append_line("return tick['open_price'] > 25000")

    return enter_trade_cond.get()


def close_long_trade_cond_basic():
    enter_trade_cond = PyCode()

    enter_trade_cond.append_line("def close_long_trade(tick):")
    enter_trade_cond.add_indent()
    enter_trade_cond.append_line("return tick['open_price'] > 30000")

    return enter_trade_cond.get()


def open_short_trade_cond_basic():
    exit_trade_cond = PyCode()

    exit_trade_cond.append_line("def open_short_trade(tick):")
    exit_trade_cond.add_indent()
    exit_trade_cond.append_line("return tick['open_price'] < 20000")

    return exit_trade_cond.get()


def close_short_trade_cond_basic():
    enter_trade_cond = PyCode()

    enter_trade_cond.append_line("def close_short_trade(tick):")
    enter_trade_cond.add_indent()
    enter_trade_cond.append_line("return tick['open_price'] < 18000")

    return enter_trade_cond.get()
