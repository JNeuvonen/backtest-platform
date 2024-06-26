export DATABASE_URI="postgresql://postgres:salasana123@localhost:5432/live_env_db_dump"

export SERVICE_PORT=8003
export ENV=DEV
export AUTH0_DOMAIN="dev-3db8jwnabrjinvzd.us.auth0.com"
export AUTH0_API_IDENTIFIER="https://dev-3db8jwnabrjinvzd.us.auth0.com/api/v2/"

export BINANCE_API_KEY=""
export BINANCE_API_SECRET=""
export LOG_SOURCE_PROGRAM="3"


cd projects 
cd analytics_server 
cd analytics_server

# npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload --port $SERVICE_PORT
npx kill-port $SERVICE_PORT && python -m main
