chmod +x ./scripts/node_link_common_js.sh
./scripts/node_link_common_js.sh


export REACT_APP_ANALYTICS_SERV_URI="http://localhost:8000/"

cd projects 
cd analytics_www
npm run start
