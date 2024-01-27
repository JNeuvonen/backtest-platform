from code_gen_template import TRAIN_TEMPLATE
from config import is_testing
from log import LogExceptionContext
from query_dataset import DatasetQuery
from query_model import Model
from query_trainjob import TrainJob
from utils import convert_val_split_str_to_arr, global_symbols, run_in_thread


class PyCodeBuilder:
    def __init__(self):
        self.lines = []

    def add_line(self, line):
        self.lines.append(line)

    def add_block(self, body_lines):
        for line in body_lines:
            self.add_line(line)

    def get_code(self):
        return "\n".join(self.lines)


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


async def start_train_loop(
    model: Model, train_job: TrainJob, batch_size=32, shuffle=True
):
    with LogExceptionContext():
        global_symbols.cleanup_globals()
        template = TRAIN_TEMPLATE

        dataset = DatasetQuery.fetch_dataset_by_id(model.dataset_id)

        if dataset is None:
            raise ValueError(f"No dataset was found for {model.dataset_id}")

        replacements = {
            "{MODEL_CLASS}": model.model_code,
            "{DATASET_NAME}": f"'{dataset.dataset_name}'",
            "{TARGET_COL}": f"'{model.target_col}'",
            "{NULL_FILL_STRATEGY}": model.null_fill_strategy,
            "{BATCH_SIZE}": batch_size,
            "{SHUFFLE}": shuffle,
            "{CRITERION_AND_OPTIMIZER}": model.optimizer_and_criterion_code,
            "{NUM_EPOCHS}": train_job.num_epochs,
            "{TRAIN_JOB_ID}": train_job.id,
            "{SAVE_MODEL_EVERY_EPOCH}": train_job.save_model_every_epoch,
            "{TRAIN_VAL_SPLIT}": convert_val_split_str_to_arr(model.validation_split),
        }

        for key, value in replacements.items():
            template = template.replace(key, str(value))

        if is_testing():
            exec(template, globals())
        else:
            run_in_thread(exec, template, globals())
