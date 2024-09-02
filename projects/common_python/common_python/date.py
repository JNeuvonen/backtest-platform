from datetime import datetime


def get_time_elapsed(start_time: datetime, end_time: datetime):
    time_difference = end_time - start_time
    total_seconds = time_difference.total_seconds()

    if total_seconds < 60:
        elapsed_time = f"{total_seconds:.2f} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        elapsed_time = f"{minutes:.2f} minutes"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        elapsed_time = f"{hours:.2f} hours"
    else:
        days = total_seconds / 86400
        elapsed_time = f"{days:.2f} days"
    return elapsed_time


def get_diff_in_ms(start_time: datetime, end_time: datetime) -> int:
    time_difference = end_time - start_time
    total_milliseconds = int(time_difference.total_seconds() * 1000)
    return total_milliseconds


def format_ms_to_human_readable(milliseconds: int) -> str:
    if milliseconds < 0:
        return "Invalid input: milliseconds cannot be negative"

    if milliseconds < 1000:
        result = f"{milliseconds} milliseconds"
    elif milliseconds < 60000:
        seconds = milliseconds / 1000
        result = f"{seconds:.2f} seconds"
    elif milliseconds < 3600000:
        minutes = milliseconds / 60000
        result = f"{minutes:.2f} minutes"
    elif milliseconds < 86400000:
        hours = milliseconds / 3600000
        result = f"{hours:.2f} hours"
    else:
        days = milliseconds / 86400000
        result = f"{days:.2f} days"
    return result


def iso_to_timestamp_ms(iso_string: str):
    dt = datetime.fromisoformat(iso_string)
    timestamp_s = dt.timestamp()
    timestamp_ms = int(timestamp_s * 1000)
    return timestamp_ms
