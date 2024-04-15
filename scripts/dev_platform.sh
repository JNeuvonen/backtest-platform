cd backtest_platform
export REACT_APP_PRED_SERVER_URI="http://34.146.92.194"
# export REACT_APP_PRED_SERVER_URI="http://localhost:8001" 
npx kill-port 8000 && npm run tauri dev
