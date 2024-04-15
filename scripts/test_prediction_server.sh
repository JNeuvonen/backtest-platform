#!/bin/bash

export TEST_RUN_PORT=3002
export SERVICE_PORT=3002
export ENV=DEV
export IS_TESTING="1"
export DATABASE_URI=postgresql://postgres:password@localhost/pred_server_integration_tests

pytest "tests/prediction_server" -s -m "acceptance"


