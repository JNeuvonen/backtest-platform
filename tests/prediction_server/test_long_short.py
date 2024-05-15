import time
import pytest

from tests.prediction_server.fixtures.long_short import (
    long_short_body_basic,
    long_short_test_enter_trade,
)
from tests.prediction_server.t_utils import Post


@pytest.mark.long_short_dev
def test_setup_sanity(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()

    Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)

    time.sleep(1000000)


@pytest.mark.acceptance
def test_longshort_enter_trade(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()
    Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)

    time.sleep(20)

    longshort_enter_trade_body = long_short_test_enter_trade()
    Post.enter_longshort_trade(fixt_create_master_acc, 1, longshort_enter_trade_body)
