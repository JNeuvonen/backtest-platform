#!/bin/bash

export APP_DATA_PATH="tests"
export ENV="PROD"
export TEST_SPEED="FAST"

pytest "tests" -m acceptance

