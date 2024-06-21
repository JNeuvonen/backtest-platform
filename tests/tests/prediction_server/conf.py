import os
from dotenv import load_dotenv

load_dotenv()


SERVICE_CODE_SOURCE_DIR = "projects/prediction_server/prediction_server/"
TEST_RUN_PORT = os.getenv("TEST_RUN_PORT", 3002)
DROP_TABLES = os.getenv("DROP_TABLES", None)
