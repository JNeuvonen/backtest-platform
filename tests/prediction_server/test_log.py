import pytest

from t_utils import Get, Post


@pytest.mark.acceptance
def test_cloud_logs_sanity(cleanup_db, create_api_key):
    Post.create_cloud_log(
        create_api_key, {"message": "hello_world", "level": "info", "source_program": 2}
    )
    Post.create_cloud_log(
        create_api_key,
        {"message": "hello_world_debug", "level": "debug", "source_program": 2},
    )
