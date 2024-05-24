import pytest
import time
from tests.analytics_server.fixtures.user import admin_user
from tests.analytics_server.http_utils import Post, Get


@pytest.mark.acceptance
def test_user_sanity(cleanup_db):
    time.sleep(5)
    user_id = Post.create_user(admin_user)
    print(user_id)


@pytest.mark.acceptance
def test_protected_endpoint():
    res = Get.call_user_root()
    print(res)
