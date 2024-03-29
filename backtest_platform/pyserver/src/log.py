import asyncio
from contextlib import contextmanager
import logging
import json
import threading
import inspect
import os
from config import append_app_data_path, is_dev

from constants import LOG_FILE


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


@contextmanager
def LogExceptionContext(
    custom_handler=None,
    logging_level=logging.ERROR,
    ui_dom_event="",
    notification_duration=5000,
    re_raise=True,
    success_log_msg="",
):
    logger = get_logger()
    if is_dev():
        stack_frame = capture_stack_frame(
            inspect.stack()[2].function, get_context_frame_params()
        )
        logger.log(
            stack_frame,
            logging.INFO,
            False,
            False,
        )
    try:
        yield
        if success_log_msg != "":
            logger.log(success_log_msg, logging.INFO)
    except Exception as e:
        if custom_handler and custom_handler(e):
            return
        stack_frame = capture_stack_frame(
            inspect.stack()[2].function, get_context_frame_params()
        )
        logger.save_exception_stackframe(stack_frame, str(e))
        logger.log(
            f"{str(e)}",
            logging_level,
            False,
            False,
            ui_dom_event,
            notification_duration,
        )
        if re_raise:
            raise


class Logger:
    def __init__(self, log_file, log_level=logging.INFO):
        if os.path.exists(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.websocket_connections = []

        # Check if the logger already has a FileHandler
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.info("Application launched")

    def add_websocket_connection(self, websocket):
        """Add a new WebSocket connection."""
        self.websocket_connections.append(websocket)

    def remove_websocket_connection(self, websocket):
        """Remove a WebSocket connection."""
        self.websocket_connections.remove(websocket)

    def build_stream_msg(
        self,
        message,
        log_level,
        display_in_ui,
        should_refetch,
        ui_dom_event,
        notification_duration,
    ):
        return {
            "msg": message,
            "log_level": log_level,
            "display": display_in_ui,
            "refetch": should_refetch,
            "dom_event": ui_dom_event,
            "notification_duration": notification_duration,
        }

    def save_exception_stackframe(self, stack_frame, error_msg):
        self.logger.error(
            f"exception was raised: {error_msg}\nstack frame: {stack_frame}"
        )

    def log(
        self,
        message,
        log_level,
        display_in_ui=False,
        should_refetch=False,
        ui_dom_event="",
        notification_duration=5000,
    ):
        """Send a message to all WebSocket connections before logging."""
        stream_msg_body = self.build_stream_msg(
            message,
            log_level,
            display_in_ui,
            should_refetch,
            ui_dom_event,
            notification_duration,
        )

        async def send_message_async(websocket):
            try:
                await websocket.send_text(json.dumps(stream_msg_body))
            except Exception as _:
                return websocket

        def send_message_sync(websocket):
            return asyncio.run(send_message_async(websocket))

        disconnected_sockets = []
        threads = []

        for websocket in self.websocket_connections:
            thread = threading.Thread(target=send_message_sync, args=(websocket,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for websocket in disconnected_sockets:
            self.remove_websocket_connection(websocket)

        if log_level == logging.INFO:
            self.info(message)
        elif log_level == logging.WARNING:
            self.warning(message)
        elif log_level == logging.ERROR:
            self.error(message)
        elif log_level == logging.DEBUG:
            self.debug(message)

    def info(self, message):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)


logger = Logger(
    append_app_data_path(LOG_FILE),
    logging.INFO,
)


def get_logger():
    return logger
