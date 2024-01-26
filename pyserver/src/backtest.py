from typing import List

from pandas.io.parquet import json
from dataset import load_data
from log import LogExceptionContext
from orm import Dataset, Model, ModelWeights, TrainJob, TrainJobQuery
from request_types import BodyRunBacktest
from utils import convert_val_split_str_to_arr
import torch
from torch.utils.data import DataLoader


def run_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        model: Model = train_job_detailed["model"]
        train_job: TrainJob = train_job_detailed["train_job"]
        epochs: List[ModelWeights] = train_job_detailed["epochs"]
        dataset: Dataset = train_job_detailed["dataset"]

        target_col = json.loads(train_job.validation_target_before_scale)
        predictions = json.loads(epochs[5]["val_predictions"])

        assert len(target_col) == len(predictions)
