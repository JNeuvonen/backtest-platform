import os

from tests.backtest_platform.t_constants import EnvTestSpeed

TEST_SPEED = os.getenv("TEST_SPEED", "FAST")  # slow and fast


def is_fast_test_mode():
    return TEST_SPEED == EnvTestSpeed.FAST
