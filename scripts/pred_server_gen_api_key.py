import os
import psycopg2
import secrets

from dotenv import load_dotenv


load_dotenv()

PRED_SERVER_DB_HOST = os.getenv("PRED_SERVER_DB_HOST", "")
PRED_SERVER_DB_USER = os.getenv("PRED_SERVER_DB_USER", "")
PRED_SERVER_DB_PASSWORD = os.getenv("PRED_SERVER_DB_PASSWORD", "")
PRED_SERVER_DB_NAME = os.getenv("PRED_SERVER_DB_NAME", "")


def generate_api_key():
    return secrets.token_urlsafe(32)


def create_api_key():
    try:
        connection = psycopg2.connect(
            dbname=PRED_SERVER_DB_NAME,
            user=PRED_SERVER_DB_USER,
            password=PRED_SERVER_DB_PASSWORD,
            host=PRED_SERVER_DB_HOST,
        )
        cursor = connection.cursor()
        api_key = generate_api_key()
        insert_query = "INSERT INTO api_keys (key) VALUES (%s) RETURNING id;"
        cursor.execute(insert_query, (api_key,))
        key_id = cursor.fetchone()[0]
        connection.commit()
        print(f"API Key created with id: {key_id}")
        print(f"Here is your API_KEY: {api_key}")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()


if __name__ == "__main__":
    if (
        PRED_SERVER_DB_HOST == ""
        or PRED_SERVER_DB_USER == ""
        or PRED_SERVER_DB_PASSWORD == ""
        or PRED_SERVER_DB_NAME == ""
    ):
        raise Exception("No valid database creds provided")
    create_api_key()
