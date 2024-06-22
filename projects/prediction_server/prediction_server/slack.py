import threading
import requests

from prediction_server.constants import LogLevel


def post_slack_message(uri, message, message_type=None):
    if uri is None or message is None:
        return

    def send_request():
        try:
            headers = {"Content-Type": "application/json"}
            # Adjust the payload according to the message type
            if LogLevel.EXCEPTION == message_type:
                payload = {
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "*Error*: " + message},
                        }
                    ]
                }
            elif LogLevel.INFO == message_type:
                payload = {
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "*Info*: " + message},
                        }
                    ]
                }
            else:
                payload = {"text": message}

            requests.post(uri, json=payload, headers=headers)
        except Exception as e:
            print(f"Failed to send request: {e}")

    thread = threading.Thread(target=send_request)
    thread.start()
