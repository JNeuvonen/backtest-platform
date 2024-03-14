import pytest
from tests.fixtures import NUM_EPOCHS_DEFAULT, create_backtest, create_train_job_basic
from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import Constants, DatasetMetadata

from tests.t_utils import Fetch, Post

import sys

sys.path.append(SERVER_SOURCE_DIR)
from query_trainjob import TrainJobQuery


@pytest.mark.acceptance
def test_fetch_model_by_name(cleanup_db, create_basic_model: DatasetMetadata):
    model = Fetch.get_model_by_name(Constants.EXAMPLE_MODEL_NAME)
    assert model["model_name"] == Constants.EXAMPLE_MODEL_NAME


@pytest.mark.acceptance
def test_create_train_job(cleanup_db, create_basic_model: DatasetMetadata):
    Post.create_train_job(Constants.EXAMPLE_MODEL_NAME, body=create_train_job_basic())


@pytest.mark.acceptance
def test_create_train_job_finished_with_status_false(
    cleanup_db, create_basic_model: DatasetMetadata
):
    id = Post.create_train_job(
        Constants.EXAMPLE_MODEL_NAME, body=create_train_job_basic()
    )

    train_job = TrainJobQuery.get_train_job(id)

    assert train_job.is_training is False


@pytest.mark.acceptance
def test_changing_is_training_status(cleanup_db, create_train_job):
    train_job = create_train_job[2]

    TrainJobQuery.set_training_status(train_job.id, True)
    assert TrainJobQuery.get_train_job(train_job.id).is_training is True

    TrainJobQuery.set_training_status(train_job.id, False)
    assert TrainJobQuery.get_train_job(train_job.id).is_training is False


@pytest.mark.acceptance
def test_route_fetch_all_metadata_by_name(cleanup_db, create_train_job):
    model = create_train_job[1]

    Post.create_train_job(Constants.EXAMPLE_MODEL_NAME, body=create_train_job_basic())
    all_metadata = Fetch.all_metadata_by_model_name(model["name"])

    assert len(all_metadata) == 2
    assert len(all_metadata[0]["weights"]) == NUM_EPOCHS_DEFAULT
    assert len(all_metadata[1]["weights"]) == NUM_EPOCHS_DEFAULT


@pytest.mark.acceptance
def test_route_create_backtest(cleanup_db, create_train_job):
    train_job = create_train_job[2]
    Post.create_model_backtest(train_job.id, create_backtest(create_train_job[0].name))
