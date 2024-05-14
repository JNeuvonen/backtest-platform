import time
import pytest

from tests.prediction_server.fixtures.long_short import long_short_body_basic
from tests.prediction_server.t_utils import Post


@pytest.mark.long_short_dev
def test_setup_sanity(cleanup_db, fixt_create_master_acc):
    long_short_body = long_short_body_basic()

    Post.create_long_short_strategy(fixt_create_master_acc, long_short_body)

    time.sleep(1000000)
