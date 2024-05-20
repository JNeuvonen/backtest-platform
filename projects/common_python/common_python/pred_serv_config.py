import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URI = os.getenv("DATABASE_URI", "")


def get_db_uri():
    if DATABASE_URI == "":
        raise Exception("No DATABASE_URI was provided")
    return DATABASE_URI
