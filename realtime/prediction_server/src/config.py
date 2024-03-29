import os
from dotenv import load_dotenv


load_dotenv()

DATABASE_URI = os.getenv(
    "DATABASE_URI", "postgresql://username:password@localhost/live_env_local_test"
)
SERVICE_PORT = os.getenv("SERVICE_PORT", "")
ENV = os.getenv("ENV", "")
LOG_FILE = "logs"


def get_db_uri():
    if DATABASE_URI == "":
        raise Exception("No DATABASE_URI was provided")
    return DATABASE_URI


def get_service_port():
    if SERVICE_PORT == "":
        raise Exception("No SERVICE_PORT was provided")
    return int(SERVICE_PORT)


def is_dev():
    if ENV == "":
        raise Exception("No ENV (PROD or DEV) was provided")

    return ENV == "DEV"
