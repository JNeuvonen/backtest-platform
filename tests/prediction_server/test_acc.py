import pytest
from t_utils import Get, Post
from fixtures.account import create_master_acc


@pytest.mark.acceptance
def test_account_sanity(cleanup_db, create_api_key):
    master_acc = create_master_acc()
    Post.create_account(create_api_key, master_acc)
    accounts = Get.fetch_accounts(create_api_key)
    assert len(accounts) == 1
