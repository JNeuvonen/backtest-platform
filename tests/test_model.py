import pytest
from tests.t_constants import Constants, DatasetMetadata

from tests.t_utils import Fetch


@pytest.mark.acceptance
def test_fetch_model_by_name(cleanup_db, create_basic_model: DatasetMetadata):
    model = Fetch.get_model_by_name(Constants.EXAMPLE_MODEL_NAME)
    assert model["name"] == Constants.EXAMPLE_MODEL_NAME
