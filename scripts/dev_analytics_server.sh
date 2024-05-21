export DATABASE_URI="postgresql+psycopg2://postgres:salasana123@localhost/pred_server_integration_tests"

export SERVICE_PORT=8001
export ENV=DEV
export AUTH0_DOMAIN="dev-3db8jwnabrjinvzd.us.auth0.com"
export AUTH0_API_IDENTIFIER="https://dev-3db8jwnabrjinvzd.us.auth0.com/api/v2/"


cd projects 
cd analytics_server 
cd analytics_server

npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload
