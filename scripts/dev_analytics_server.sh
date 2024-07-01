#!/bin/bash

export DATABASE_URI="postgresql://postgres:password@localhost:5432/algorithmic_trading_local"
export SERVICE_PORT=8003
export ENV=DEV
export AUTH0_DOMAIN="dev-3db8jwnabrjinvzd.us.auth0.com"
export AUTH0_API_IDENTIFIER="https://dev-3db8jwnabrjinvzd.us.auth0.com/api/v2/"

# Check if BINANCE_API_KEY is set
if [ -z "${BINANCE_API_KEY}" ]; then
  echo "Error: BINANCE_API_KEY is not set."
  exit 1
fi

# Check if BINANCE_API_SECRET is set
if [ -z "${BINANCE_API_SECRET}" ]; then
  echo "Error: BINANCE_API_SECRET is not set."
  exit 1
fi

export BINANCE_API_KEY=${BINANCE_API_KEY}
export BINANCE_API_SECRET=${BINANCE_API_SECRET}
export LOG_SOURCE_PROGRAM="3"

cd projects 
cd analytics_server 
cd analytics_server

# npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload --port $SERVICE_PORT
npx kill-port $SERVICE_PORT && python -m main
