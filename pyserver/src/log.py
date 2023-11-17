import logging
import os


class Logger:
    def __init__(self, log_file, log_level=logging.INFO):
        if os.path.exists(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Check if the logger already has a FileHandler
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.info("Application booted")

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
