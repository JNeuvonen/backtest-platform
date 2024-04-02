import pytest

from t_utils import Get, Post


@pytest.mark.acceptance
def test_cloud_logs_sanity(cleanup_db, create_api_key):
    Post.create_cloud_log(create_api_key, {"message": "hello_world", "level": "info"})
    Post.create_cloud_log(
        create_api_key, {"message": "hello_world_debug", "level": "debug"}
    )

    created_logs = Get.fetch_cloud_logs(create_api_key)

    assert (
        len(created_logs) == 2
    ), "test_cloud_logs_sanity: logs were not created succesfully"
