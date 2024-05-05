#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <database_uri>"
  exit 1
fi

export DATABASE_URL="$1"

cd realtime
cd prediction_server
ALEMBIC_CONFIG=./alembic.ini alembic upgrade head
