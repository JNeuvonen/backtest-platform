#!/bin/bash

export APP_DATA_PATH="tests"
export ENV="PROD"
export TEST_SPEED="FAST"
export IS_TESTING="1"

pytest "tests/test_model.py" -m acceptance


