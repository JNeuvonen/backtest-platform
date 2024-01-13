import os

APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
ENV = os.getenv("ENV", "")


def is_prod():
    return ENV == "PROD"


def is_dev():
    return ENV == "DEV"


def append_app_data_path(appended_path):
    return os.path.join(APP_DATA_PATH, appended_path)
