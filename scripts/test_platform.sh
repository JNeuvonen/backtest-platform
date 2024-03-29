#!/bin/bash

export APP_DATA_PATH="tests/backtest_platform"
export ENV="PROD"
export TEST_SPEED="FAST"
export IS_TESTING="1"

pytest "tests/backtest_platform" -m acceptance


