from pyserver.src.code_gen import PyCode
from tests.t_utils import create_train_job_body


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
    helper.append_line("criterion = nn.MSELoss()")
    helper.append_line(
        "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)"
    )

    return helper.get()


def create_train_job_basic():
    enter_trade_criteria = PyCode()
    enter_trade_criteria.append_line("prediction = get_prediction()")
    enter_trade_criteria.append_line("return prediction > 1.01")

    exit_trade_criteria = PyCode()
    exit_trade_criteria.append_line("prediction = get_prediction()")
    exit_trade_criteria.append_line("return prediction < 0.99")

    body = create_train_job_body(
        num_epochs=100,
        save_model_after_every_epoch=True,
        backtest_on_val_set=True,
        enter_trade_criteria=enter_trade_criteria.get(),
        exit_trade_criteria=exit_trade_criteria.get(),
    )
    return body
