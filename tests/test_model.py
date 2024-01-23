import pytest
from tests.fixtures import create_train_job_basic
from tests.t_constants import Constants, DatasetMetadata

from tests.t_utils import Fetch, Post


@pytest.mark.acceptance
def test_fetch_model_by_name(cleanup_db, create_basic_model: DatasetMetadata):
    model = Fetch.get_model_by_name(Constants.EXAMPLE_MODEL_NAME)
    assert model["model_name"] == Constants.EXAMPLE_MODEL_NAME


@pytest.mark.acceptance
def test_create_train_job(cleanup_db, create_basic_model: DatasetMetadata):
    Post.create_train_job(Constants.EXAMPLE_MODEL_NAME, body=create_train_job_basic())
