import os
from dotenv import load_dotenv


load_dotenv()


SERVICE_PORT = os.getenv("SERVICE_PORT", "")


def get_service_port():
    if SERVICE_PORT == "":
        raise Exception("No SERVICE_PORT was provided")
    return int(SERVICE_PORT)
