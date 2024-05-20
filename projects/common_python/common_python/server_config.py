import os
from dotenv import load_dotenv


load_dotenv()


SERVICE_PORT = os.getenv("SERVICE_PORT", "")
ENV = os.getenv("ENV", "")


def get_service_port():
    if SERVICE_PORT == "":
        raise Exception("No SERVICE_PORT was provided")
    return int(SERVICE_PORT)


def is_dev():
    if ENV == "":
        raise Exception("No ENV (PROD or DEV) was provided")

    return ENV == "DEV"
