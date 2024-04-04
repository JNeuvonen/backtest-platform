#!/bin/bash

cd realtime
cd trading_client

TEST_FLAG=""

if [[ ! -z $1 ]] && [[ $1 == "--test" ]]; then
    TEST_FLAG="-run $2"
fi

go test -count=1 -v ./src $TEST_FLAG
