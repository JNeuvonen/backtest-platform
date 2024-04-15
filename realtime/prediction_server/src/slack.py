import requests
import threading


def post_message(uri, message):
    def send_request():
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"text": message}
            requests.post(uri, json=payload, headers=headers)
        except Exception as e:
            print(f"Failed to send request: {e}")

    thread = threading.Thread(target=send_request)
    thread.start()
