import pytest
from tests.analytics_server.fixtures.user import admin_user
from tests.analytics_server.http_utils import Post, Get


@pytest.mark.acceptance
def test_user_sanity(cleanup_db):
    user_id = Post.create_user(admin_user)
    print("hello world")


@pytest.mark.acceptance
def test_protected_endpoint():
    Get.call_user_root()
