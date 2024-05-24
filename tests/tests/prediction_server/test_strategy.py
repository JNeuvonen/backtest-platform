import pytest
import time
from t_utils import Get, Post
from fixtures.strategy import (
    gen_test_case_dump_body,
    strategy_simple_1,
)


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, create_api_key):
    body = strategy_simple_1()
    Post.create_strategy(create_api_key, body=body)
    strategies = Get.fetch_strategies(create_api_key)
    assert len(strategies) == 1, "test_setup_sanity: no strategy was created"


@pytest.mark.slow
def test_strategy_polling(cleanup_db, create_api_key):
    Post.create_strategy(create_api_key, body=gen_test_case_dump_body())
    time.sleep(1000)


@pytest.mark.debug_live_env
def test_debug_live_env():
    time.sleep(1000)
