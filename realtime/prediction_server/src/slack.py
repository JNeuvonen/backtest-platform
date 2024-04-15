import requests
import threading


def post_message(uri, message):
    def send_request():
        headers = {"Content-Type": "application/json"}
        payload = {"text": message}
        requests.post(uri, json=payload, headers=headers)

    thread = threading.Thread(target=send_request)
    thread.start()
