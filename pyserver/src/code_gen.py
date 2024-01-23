from db_objects import ModelObject, TrainJobObject
from log import LogExceptionContext
from orm import DatasetQuery, Model, TrainJob, ModelWeightsQuery
from utils import global_symbols

# Used by generated python code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_train_data
import pickle
from config import append_app_data_path


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


class CodeGen:
    def __init__(self):
        self.code = PyCode()
        self.global_manager = global_symbols

    def train_job(self, model: Model, train_job: TrainJob, batch_size=32, shuffle=True):
        with LogExceptionContext():
            global_symbols.cleanup_globals()

            dataset = DatasetQuery.fetch_dataset_by_id(model.dataset_id)

            if dataset is None:
                raise ValueError(f"No dataset was found for {model.dataset_id}")

            exec(model.model_code, globals())
            exec(model.optimizer_and_criterion_code, globals())

            self.code.reset_code()

            self.code.append_line("def train():")
            self.code.add_indent()

            self.code.append_line(
                f"x_train, y_train = load_train_data('{dataset.dataset_name}', '{model.target_col}', {model.null_fill_strategy})"
            )
            self.code.append_line("dataset = TensorDataset(x_train, y_train)")

            self.code.append_line(
                f"train_loader = DataLoader(dataset, batch_size={batch_size}, shuffle={shuffle})"
            )

            self.code.append_line("model = Model(x_train.shape[1])")
            self.code.append_line(
                f"criterion, optimizer = get_criterion_and_optimizer(model)"
            )

            self.code.append_line(
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            )

            self.code.append_line("model.to(device)")

            self.code.append_line(f"for epoch in range({train_job.num_epochs}):")
            self.code.add_indent()

            self.code.append_line("for inputs, labels in train_loader:")
            self.code.add_indent()
            self.code.append_line(
                "inputs, labels = inputs.to(device), labels.to(device)"
            )
            self.code.append_line("outputs = model(inputs)")
            self.code.append_line("l1_lambda = 0.01")
            self.code.append_line(
                "l1_norm = sum(p.abs().sum() for p in model.parameters())"
            )
            self.code.append_line(
                "loss = criterion(outputs, labels) + l1_lambda * l1_norm"
            )
            self.code.append_line("optimizer.zero_grad()")
            self.code.append_line("loss.backward()")
            self.code.append_line("optimizer.step()")
            self.code.reduce_indent()

            self.code.append_line(
                "model_weights_bytes = pickle.dumps(model.state_dict())"
            )
            self.code.append_line(
                f"ModelWeightsQuery.create_model_weights_entry('{train_job.id}', epoch, model_weights_bytes)"
            )

            self.code.reset_indent()
            self.code.append_line("train()")

            exec(self.code.get(), globals())
