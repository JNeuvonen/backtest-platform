import os
from dotenv import load_dotenv

load_dotenv()


TEST_RUN_PORT = os.getenv("TEST_RUN_PORT", 3002)
DROP_TABLES = os.getenv("DROP_TABLES", None)
