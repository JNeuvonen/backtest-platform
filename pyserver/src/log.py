import logging
import os

from constants import LOG_FILE


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

    async def log(self, message, log_level, display_in_ui=False, should_refetch=False):
        """Send a message to all WebSocket connections before logging."""
        disconnected_sockets = []
        refetch_indicator = "[REFETCH]" if should_refetch else ""

        for websocket in self.websocket_connections:
            try:
                if display_in_ui and log_level == logging.ERROR:
                    message = f"[UI-ERROR]{refetch_indicator}:{message}"
                elif display_in_ui and log_level == logging.INFO:
                    message = f"[UI-INFO]{refetch_indicator}:{message}"
                elif display_in_ui and log_level == logging.DEBUG:
                    message = f"[UI-DEBUG]{refetch_indicator}:{message}"
                elif display_in_ui and log_level == logging.WARNING:
                    message = f"[UI-WARNING]{refetch_indicator}:{message}"
                await websocket.send_text(message)
            except Exception as _:
                disconnected_sockets.append(websocket)

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


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")


logger = Logger(
    os.path.join(APP_DATA_PATH, LOG_FILE),
    logging.INFO,
)


def get_logger():
    return logger
