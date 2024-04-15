class CodeTemplates:
    FETCH_DATASOURCES = """
{FETCH_DATASOURCES_FUNC}
fetched_data = fetch_datasources() 
"""

    DATA_TRANSFORMATIONS = """
{DATA_TRANSFORMATIONS_FUNC}
transformed_data = make_data_transformations(fetched_data)
"""
    GEN_TRADE_DECISIONS = """
{ENTER_TRADE_FUNC}
{EXIT_TRADE_FUNC}

should_enter_trade = get_enter_trade_decision(transformed_data)
should_exit_trade = get_exit_trade_decision(transformed_data)
"""


class PyCode:
    def __init__(self):
        self.indent_level = 0
        self.code = ""

    def append_line(self, line: str):
        new_line = "\t" * self.indent_level + line + "\n"
        self.code += new_line

    def stringify(self, var):
        return f'"{var}"'

    def add_indent(self):
        self.indent_level += 1

    def add_pytorch(self):
        self.append_line("import torch")
        self.append_line("import torch.nn as nn")
        self.append_line("import torch.optim as optim")
        self.append_line("from torch.utils.data import TensorDataset, DataLoader")

    def reset_indent(self):
        self.indent_level = 0

    def reduce_indent(self):
        if self.indent_level > 0:
            self.indent_level -= 1

    def get(self):
        return self.code

    def reset_code(self):
        self.code = ""
        self.indent_level = 0

    def add_block(self, block_code):
        block_code = "\t" * self.indent_level + block_code.replace(
            "\n", "\n" + "\t" * self.indent_level
        )
        self.code += block_code + "\n"
