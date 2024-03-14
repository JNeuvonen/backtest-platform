import time
import pytest

from tests.fixtures import (
    create_manual_backtest,
    enter_trade_cond_basic,
    exit_trade_cond_basic,
)
from tests.t_utils import Fetch, Post


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_btc_small_1h):
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    backtest_body = create_manual_backtest(
        dataset["id"], enter_trade_cond_basic(), exit_trade_cond_basic(), True
    )

    Post.create_manual_backtest(backtest_body)
