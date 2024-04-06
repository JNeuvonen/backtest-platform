export DATABASE_URI=postgresql://username:password@localhost/live_env_local_test
export SERVICE_PORT=8000
export ENV=DEV
# npx kill-port $SERVICE_PORT && python realtime/prediction_server/src/main.py
cd realtime
cd prediction_server
cd src

npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload

