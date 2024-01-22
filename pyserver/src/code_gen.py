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

    def train_job(self, model, train_job):
        pass
