#!/bin/bash

usage() {
  echo "Usage: $0 -k <api_key> -s <api_secret>"
  exit 1
}

API_KEY=""
API_SECRET=""

while getopts 'k:s:' flag; do
  case "${flag}" in
    k) API_KEY="${OPTARG}" ;;
    s) API_SECRET="${OPTARG}" ;;
    *) usage ;;
  esac
done

if [ -z "$API_KEY" ] || [ -z "$API_SECRET" ]; then
  usage
fi

cd scripts

export API_KEY="$API_KEY"
export API_SECRET="$API_SECRET"

python close_all_positions.py
