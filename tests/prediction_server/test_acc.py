import pytest
from t_utils import Get, Post
from fixtures.account import Accounts, create_master_acc


@pytest.mark.acceptance
def test_account_sanity(cleanup_db, create_api_key):
    master_acc = create_master_acc()
    Post.create_account(create_api_key, master_acc)
    accounts = Get.fetch_accounts(create_api_key)
    assert len(accounts) == 1

    account_by_name = Get.fetch_account_by_name(create_api_key, Accounts.MASTER_ACC)
    assert account_by_name is not None, "Fetching account by name was not succesful"
