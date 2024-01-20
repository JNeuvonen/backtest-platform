from tests.t_utils import CodeHelper


def linear_model_basic():
    helper = CodeHelper()
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
    helper = CodeHelper()
    helper.append_line("criterion = nn.MSELoss()")
    helper.append_line(
        "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)"
    )

    return helper.get()
