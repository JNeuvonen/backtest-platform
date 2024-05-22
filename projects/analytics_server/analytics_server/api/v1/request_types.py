from pydantic import BaseModel


class BodyCreateuser(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

    access_level: int