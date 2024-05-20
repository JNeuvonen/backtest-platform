export DATABASE_URI="postgresql+psycopg2://jarnoneuvonen:password@localhost/pred_server_integration_tests"

export SERVICE_PORT=8001
export ENV=DEV


cd projects 
cd analytics_server 
cd analytics_server

npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload
