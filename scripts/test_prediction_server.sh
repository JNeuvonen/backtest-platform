#!/bin/bash

export TEST_RUN_PORT=3002
export SERVICE_PORT=3002
export ENV=DEV
export IS_TESTING="1"

pytest "tests/prediction_server" -s -m acceptance


