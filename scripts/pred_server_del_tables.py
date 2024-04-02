import sys
import os
from dotenv import load_dotenv

load_dotenv()

PATH_TO_PRED_SERVER = "realtime/prediction_server/src"

sys.path.append(PATH_TO_PRED_SERVER)

from orm import drop_tables, engine


def confirm_action(env):
    confirmation = input(
        f"Are you sure you want to delete all database tables in {env} environment? This action cannot be undone. Type 'DEL_{env}' to confirm: "
    )
    return confirmation.strip().upper() == f"DEL_{env}"


def main():
    env = os.getenv("ENV")
    if env not in ["DEV", "PROD"]:
        print("Invalid environment. Please set ENV variable to either 'DEV' or 'PROD'.")
        return

    if confirm_action(env):
        drop_tables(engine)
        print(f"All database tables have been deleted in {env} environment.")
    else:
        print("Action canceled.")


if __name__ == "__main__":
    main()
