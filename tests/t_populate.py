import requests

from tests.t_constants import URL, DatasetMetadata
from tests.t_context import t_file


def t_upload_dataset(dataset: DatasetMetadata):
    with t_file(dataset.path) as file:
        response = requests.post(
            URL.t_get_upload_dataset_url(dataset.name, dataset.timeseries_col),
            files={"file": (dataset.path, file)},
        )
        return response
