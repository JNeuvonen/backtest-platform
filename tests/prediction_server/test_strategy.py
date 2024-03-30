import pytest
from utils import Get, Post
from fixtures.strategy import strategy_simple_1


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    Post.create_strategy(body=strategy_simple_1())
    strategies = Get.fetch_strategies()
    assert len(strategies) == 1, "test_setup_sanity: no strategy was created"
