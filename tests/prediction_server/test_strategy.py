import pytest
import time
from t_utils import Get, Post
from fixtures.strategy import strategy_simple_1


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    Post.create_strategy(body=strategy_simple_1())
    strategies = Get.fetch_strategies()
    assert len(strategies) == 1, "test_setup_sanity: no strategy was created"


@pytest.mark.slow
def test_strategy_polling(cleanup_db):
    Post.create_strategy(body=strategy_simple_1())
    time.sleep(10)
