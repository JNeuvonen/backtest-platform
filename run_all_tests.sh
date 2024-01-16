#!/bin/bash

export APP_DATA_PATH="tests"
export ENV="PROD"

pytest "tests" -m acceptance

