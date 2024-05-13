export DATABASE_URI="postgresql://postgres:password@localhost:5432/live_env_local_test"
# export DATABASE_URI=postgresql://postgres:HJyGHt67hRn8Ru@35.187.213.213:5432/prediction_server
export SERVICE_PORT=8001
export ENV=DEV
export AUTO_WHITELISTED_IP="87.94.138.211"

# npx kill-port $SERVICE_PORT && python realtime/prediction_server/src/main.py
cd realtime
cd prediction_server
cd src

npx kill-port $SERVICE_PORT && python -m uvicorn main:app --reload

