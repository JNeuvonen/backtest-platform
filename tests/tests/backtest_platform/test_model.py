import pytest
import time
from tests.backtest_platform.fixtures import (
    NUM_EPOCHS_DEFAULT,
    create_backtest,
    create_train_job_basic,
)
from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR
from tests.backtest_platform.t_constants import Constants, DatasetMetadata

from tests.backtest_platform.t_utils import Fetch, Post

import sys

sys.path.append(SERVER_SOURCE_DIR)
from query_trainjob import TrainJobQuery


@pytest.mark.acceptance
def test_fetch_model_by_name(cleanup_db, create_basic_model):
    model_id = create_basic_model[2]
    model = Fetch.get_model_by_id(model_id)
    assert model["model_name"] == Constants.EXAMPLE_MODEL_NAME


@pytest.mark.acceptance
def test_create_train_job(cleanup_db, create_basic_model: DatasetMetadata):
    model_id = create_basic_model[2]
    Post.create_train_job(model_id, body=create_train_job_basic())


@pytest.mark.acceptance
def test_create_train_job_finished_with_status_false(
    cleanup_db, create_basic_model: DatasetMetadata
):
    model_id = create_basic_model[2]
    id = Post.create_train_job(model_id, body=create_train_job_basic())

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
    model_id = create_train_job[3]

    Post.create_train_job(model_id, body=create_train_job_basic())
    all_metadata = Fetch.all_metadata_by_model_id(model_id)

    assert len(all_metadata) == 2
    assert len(all_metadata[0]["weights"]) == NUM_EPOCHS_DEFAULT
    assert len(all_metadata[1]["weights"]) == NUM_EPOCHS_DEFAULT


@pytest.mark.acceptance
def test_route_create_backtest(cleanup_db, create_train_job):
    train_job = create_train_job[2]
    Post.create_model_backtest(train_job.id, create_backtest(create_train_job[0].name))
