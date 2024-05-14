export DATABASE_URI="postgresql+psycopg2://jarnoneuvonen:password@localhost/pred_server_integration_tests"
# export DATABASE_URI=postgresql://postgres:HJyGHt67hRn8Ru@35.187.213.213:5432/prediction_server
export SERVICE_PORT=8001
export ENV=DEV
export AUTO_WHITELISTED_IP="185.11.209.117"

# npx kill-port $SERVICE_PORT && python realtime/prediction_server/src/main.py
cd realtime
cd prediction_server
cd src

npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload

