from tests.t_utils import create_backtest_body, create_train_job_body

from tests.t_conf import SERVER_SOURCE_DIR
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


def create_backtest():
    code_trade_criteria = PyCode()
    code_trade_criteria.append_line("def get_enter_trade_criteria(prediction):")
    code_trade_criteria.add_indent()
    code_trade_criteria.append_line("return prediction > 1.01")
    code_trade_criteria.reduce_indent()
    code_trade_criteria.append_line("def get_exit_trade_criteria(prediction):")
    code_trade_criteria.add_indent()
    code_trade_criteria.append_line("return prediction < 0.99")
    return create_backtest_body(
        price_column="Open price",
        epoch_nr=3,
        enter_trade_criteria=code_trade_criteria.get(),
    )
