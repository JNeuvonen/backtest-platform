import os
from dotenv import load_dotenv

load_dotenv()


SERVICE_CODE_SOURCE_DIR = "realtime/prediction_server/src"
TEST_RUN_PORT = os.getenv("TEST_RUN_PORT", 3002)
