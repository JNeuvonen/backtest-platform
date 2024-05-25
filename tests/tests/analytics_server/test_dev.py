import time
import pytest
from tests.analytics_server.http_utils import Post
from tests.analytics_server.fixtures.user import admin_user


@pytest.mark.dev
def test_dev(cleanup_db, create_ls_strategy, create_directional_strategy):
    # just start the server so debugging servers background processes is easier
    time.sleep(10)
    user_id = Post.create_user(admin_user)
    time.sleep(10000)
