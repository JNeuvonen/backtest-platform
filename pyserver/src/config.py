import os

APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
ENV = os.getenv("ENV", "")


def is_prod():
    return ENV == "PROD"


def is_dev():
    return ENV == "DEV"
