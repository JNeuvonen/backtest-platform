#!/bin/bash

usage() {
  echo "Usage: $0 -d <database_uri> -m <migration_message>"
  exit 1
}

DATABASE_URI=""
MIGRATION_MESSAGE=""

while getopts 'd:m:' flag; do
  case "${flag}" in
    d) DATABASE_URI="${OPTARG}" ;;
    m) MIGRATION_MESSAGE="${OPTARG}" ;;
    *) usage ;;
  esac
done

if [ -z "$DATABASE_URI" ] || [ -z "$MIGRATION_MESSAGE" ]; then
  usage
fi

export DATABASE_URL="$DATABASE_URI"

cd realtime/prediction_server

ALEMBIC_CONFIG=./alembic.ini alembic revision --autogenerate -m "$MIGRATION_MESSAGE"
