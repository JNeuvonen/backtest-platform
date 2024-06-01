from pydantic import BaseModel


class BinanceUserAsset(BaseModel):
    asset: str
    borrowed: float
    free: float
    interest: float
    locked: float
    netAsset: float
