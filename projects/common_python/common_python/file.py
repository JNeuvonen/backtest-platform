import os


def remove_file_if_exists(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)
        return True
    else:
        return False
