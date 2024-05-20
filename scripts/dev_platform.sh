cd projects
cd backtest_platform

#no trailing slash!
export REACT_APP_PRED_SERVER_URI="http://34.97.121.158"

# export REACT_APP_PRED_SERVER_URI="http://localhost:8001" 
npx kill-port 8000 && npm run tauri dev
