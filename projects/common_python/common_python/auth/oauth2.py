import requests
import os
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from fastapi import Depends, HTTPException
from pydantic import BaseModel

from common_python.pred_serv_models.user import UserQuery

load_dotenv()


AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "")
API_IDENTIFIER = os.getenv("AUTH0_API_IDENTIFIER", "")
ALGORITHMS = ["RS256"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_public_key():
    if AUTH0_DOMAIN == "":
        raise Exception("No AUTH0_DOMAIN provided")

    if API_IDENTIFIER == "":
        raise Exception("No API_IDENTIFIER provided")

    jsonurl = requests.get(f"https://{AUTH0_DOMAIN}/.well-known/jwks.json")
    jwks = jsonurl.json()
    return jwks["keys"][0]


def decode_jwt(token: str):
    try:
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = get_public_key()
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=API_IDENTIFIER,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
        return payload
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_info(token: str):
    try:
        userinfo_url = f"https://{AUTH0_DOMAIN}/userinfo"
        headers = {
            "Authorization": f"Bearer {token}",
        }
        response = requests.get(userinfo_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Could not retrieve user info",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def verify_access_token(token: str = Depends(oauth2_scheme)):
    decode_jwt(token)
    return token


def get_user(token: str = Depends(verify_access_token)):
    user_info = get_user_info(token)
    email = user_info["email"]
    user = UserQuery.get_user_by_email(email)

    if user is None:
        raise Exception("No user found")

    return user
