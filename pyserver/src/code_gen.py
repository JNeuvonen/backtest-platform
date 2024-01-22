from dataset import load_train_data
from db import DatasetUtils
from db_objects import ModelObject, TrainJobObject


class PyCode:
    def __init__(self):
        self.indent_level = 0
        self.code = ""

    def append_line(self, line: str):
        new_line = "    " * self.indent_level + line + "\n"
        self.code += new_line

    def add_indent(self):
        self.indent_level += 1

    def reset_indent(self):
        self.indent_level = 0

    def reduce_indent(self):
        if self.indent_level > 0:
            self.indent_level -= 1

    def get(self):
        return self.code


class CodeGen:
    def __init__(self):
        self.code = PyCode()

    def train_job(self, model: ModelObject, train_job: TrainJobObject):
        model_code = model.model
        dataset = DatasetUtils.fetch_dataset_by_id(model.dataset_id)

        pass
