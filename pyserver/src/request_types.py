from typing import List
from pydantic import BaseModel


class ModelData(BaseModel):
    name: str
    target_col: str
    drop_cols: List[str]
    null_fill_strategy: str
    model: str
    hyper_params_and_optimizer_code: str
    validation_split: List[int]
