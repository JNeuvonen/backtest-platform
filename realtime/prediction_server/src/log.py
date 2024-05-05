import inspect
import logging
from contextlib import contextmanager
import os

from config import is_dev, LOG_FILE
from logging.handlers import RotatingFileHandler

from schema.cloudlog import create_log
from constants import LogLevel


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
    def __init__(
        self,
        log_file,
        log_level=logging.INFO,
        max_size=10 * 1024 * 1024,
        backup_count=5,
    ):
        if os.path.exists(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        rotating_handler = RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
        rotating_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        rotating_handler.setFormatter(formatter)

        if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
            self.logger.addHandler(rotating_handler)

    def log_exception_stackframe(self, stack_frame, error_msg):
        self.error(f"exception was raised: {error_msg}\nstack frame: {stack_frame}")

    def info(self, message):
        create_log(msg=message, level=LogLevel.INFO)

    def warning(self, message):
        create_log(msg=message, level=LogLevel.WARNING)

    def error(self, message):
        create_log(msg=message, level=LogLevel.EXCEPTION)

    def debug(self, message):
        create_log(msg=message, level=LogLevel.DEBUG)


@contextmanager
def LogExceptionContext(
    custom_handler=None,
    re_raise=True,
    success_log_msg="",
):
    logger = get_logger()
    if is_dev():
        stack_frame = capture_stack_frame(
            inspect.stack()[2].function, get_context_frame_params()
        )

        logger.info(stack_frame)
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


logger = Logger(
    LOG_FILE,
    logging.INFO,
)


def get_logger():
    return logger
