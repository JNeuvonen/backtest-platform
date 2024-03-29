import pytest
import requests

from tests.backtest_platform.t_constants import URL


@pytest.mark.acceptance
def test_system_health():
    response = requests.get(URL.BASE_URL)
    assert response.status_code == 200
