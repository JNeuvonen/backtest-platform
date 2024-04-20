from typing import Optional

from fastapi import Query


def convert_to_bool(value: Optional[str] = Query(None)) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() in ["true", "1", "yes"]
