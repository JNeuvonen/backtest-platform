import os
import socket
from dotenv import load_dotenv


load_dotenv()

hostname = socket.gethostname()

DATABASE_URI = os.getenv("DATABASE_URI", "")
SERVICE_PORT = os.getenv("SERVICE_PORT", "")
AUTO_WHITELISTED_IP = os.getenv("AUTO_WHITELISTED_IP", "")
ENV = os.getenv("ENV", "")
LOG_FILE = "logs"


def get_auto_whitelisted_ip():
    if AUTO_WHITELISTED_IP == "":
        raise Exception("No whitelisted IP was provided")
    return AUTO_WHITELISTED_IP


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
