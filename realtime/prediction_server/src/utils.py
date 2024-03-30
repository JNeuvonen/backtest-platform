import threading


def run_in_thread(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    return thread


def replace_placeholders_on_code_templ(code, replacements):
    for key, value in replacements.items():
        code = code.replace(key, str(value))
    return code
