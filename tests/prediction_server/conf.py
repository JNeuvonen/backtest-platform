import os
from dotenv import load_dotenv

load_dotenv()


TEST_RUN_PORT = os.getenv("TEST_RUN_PORT", 3002)
