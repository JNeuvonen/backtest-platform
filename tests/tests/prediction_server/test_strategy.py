import pytest
import time
from t_utils import Get, Post
from fixtures.strategy import (
    gen_test_case_dump_body,
    strategy_simple_1,
)
from tests.prediction_server.fixtures.strategy_group import test_case_dump
from prediction_server.strategy import refresh_adaptive_strategy_group
from common_python.pred_serv_models.strategy_group import StrategyGroupQuery


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, create_api_key):
    body = strategy_simple_1()
    Post.create_strategy(create_api_key, body=body)
    strategies = Get.fetch_strategies(create_api_key)
    assert len(strategies) == 1, "test_setup_sanity: no strategy was created"


@pytest.mark.acceptance
def test_creating_strategy_group(cleanup_db, create_api_key):
    id = Post.create_strategy_group(create_api_key, body=test_case_dump)
    time.sleep(100000)
    print(id)


@pytest.mark.slow
def test_strategy_polling(cleanup_db, create_api_key):
    Post.create_strategy(create_api_key, body=gen_test_case_dump_body())
    time.sleep(1000)


@pytest.mark.debug_live_env
def test_debug_live_env():
    time.sleep(1000000)


@pytest.mark.dev
def test_adaptive_group_refresh(cleanup_db, create_api_key):
    time.sleep(5)
    id = Post.create_strategy_group(create_api_key, body=test_case_dump)
    strategy_group = StrategyGroupQuery.get_by_id(int(id))
    refresh_adaptive_strategy_group(strategy_group)
