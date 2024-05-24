import time
import pytest


@pytest.mark.dev
def test_dev():
    # just start the server so debugging servers background processes is easier
    time.sleep(10000)
