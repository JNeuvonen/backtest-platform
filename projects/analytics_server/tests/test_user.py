import pytest
from projects.analytics_server.tests.fixtures.user import admin_user
from projects.analytics_server.tests.http_utils import Post


@pytest.mark.acceptance
def test_user_sanity(cleanup_db):
    user_id = Post.create_user(admin_user)
    print(user_id)
