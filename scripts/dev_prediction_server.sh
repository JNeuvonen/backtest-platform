export DATABASE_URI=postgresql://username:password@localhost/live_env_local_test
export SERVICE_PORT=3001
export ENV=DEV
npx kill-port $SERVICE_PORT && python realtime/prediction_server/src/main.py
