import pytest
from utils import Post
from fixtures.strategy import strategy_simple_1


@pytest.mark.acceptance
def test_setup_sanity():
    res = Post.create_strategy(body=strategy_simple_1())
    print(res)
