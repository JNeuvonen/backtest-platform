import os
import inspect
from dotenv import load_dotenv
from contextlib import contextmanager

from common_python.server_config import is_dev
from common_python.pred_serv_models.cloudlog import create_log
from common_python.constants import LogLevel, LogSourceProgram

load_dotenv()


def capture_stack_frame(func_name, params):
    param_str = (
        ", ".join(f"{key}={value}" for key, value in params.items())
        if params
        else "none"
    )
    return f"function {func_name} called with parameters: {param_str}"


def get_frame_params(frame):
    params = inspect.getargvalues(frame)
    return {arg: params.locals[arg] for arg in params.args if arg != "self"}


def get_context_frame_params():
    frame = inspect.stack()[3].frame
    return get_frame_params(frame)


class Logger:
    def __init__(self, source_program):
        self.source_program = source_program

    def log_exception_stackframe(self, stack_frame, error_msg):
        self.error(f"exception was raised: {error_msg}\nstack frame: {stack_frame}")

    def info(self, message):
        create_log(msg=message, level=LogLevel.INFO, source_program=self.source_program)

    def warning(self, message):
        create_log(
            msg=message, level=LogLevel.WARNING, source_program=self.source_program
        )

    def error(self, message):
        create_log(
            msg=message, level=LogLevel.EXCEPTION, source_program=self.source_program
        )

    def debug(self, message):
        create_log(
            msg=message, level=LogLevel.DEBUG, source_program=self.source_program
        )


@contextmanager
def LogExceptionContext(
    custom_handler=None,
    re_raise=True,
    success_log_msg="",
):
    logger = get_logger()
    try:
        yield
        if success_log_msg != "":
            logger.info(success_log_msg)
    except Exception as e:
        if custom_handler and custom_handler(e):
            return
        stack_frame = capture_stack_frame(
            inspect.stack()[2].function, get_context_frame_params()
        )
        logger.log_exception_stackframe(stack_frame, str(e))
        if re_raise:
            raise


LOG_SOURCE_PROGRAM = int(os.getenv("LOG_SOURCE_PROGRAM", LogSourceProgram.PRED_SERVER))

logger = Logger(LOG_SOURCE_PROGRAM)


def get_logger():
    return logger
