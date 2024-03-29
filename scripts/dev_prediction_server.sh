export DATABASE_URI=postgresql://username:password@localhost/live_env_local_test
npx kill-port 8080 && python realtime/prediction_server/src/main.py
