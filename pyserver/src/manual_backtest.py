from typing import Dict, List
from dataset import read_dataset_to_mem
from model_backtest import Positions, Direction
from query_dataset import DatasetQuery
from request_types import BodyCreateManualBacktest


def run_manual_backtest(backtestInfo: BodyCreateManualBacktest):
    dataset = DatasetQuery.fetch_dataset_by_id(backtestInfo.dataset_id)
    dataset_df = read_dataset_to_mem(dataset.dataset_name)
    print(dataset_df)


class ManualBacktest:
    def __init__(
        self,
        start_balance: float,
        fees_perc: float,
        slippage_perc: float,
        enter_and_exit_criteria_placeholders: Dict,
    ) -> None:
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance, 1 - (fees_perc / 100), 1 - (slippage_perc / 100)
        )
        self.history: List = []
