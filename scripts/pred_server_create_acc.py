import sys

PATH_TO_TESTS_MAIN_DIR = "tests/prediction_server"
PATH_TO_PRED_SERVER = "realtime/prediction_server/src"

sys.path.append(PATH_TO_TESTS_MAIN_DIR)
sys.path.append(PATH_TO_PRED_SERVER)

from fixtures.account import create_master_acc
from orm import Session
from schema.account import AccountQuery

master_acc = create_master_acc()


strategy_id = AccountQuery.create_entry(master_acc)

print(f"Created account with ID: {strategy_id}")