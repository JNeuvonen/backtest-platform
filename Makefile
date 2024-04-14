.PHONY: build-pred-service-container

build-pred-service-container:
	docker buildx build --platform linux/amd64 -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=$(DATABASE_URI) . --load

post-test-strategy:
	python scripts/pred_server_post_strategy.py

dev-platform:
	./scripts/dev_platform.sh

loc:
	./scripts/loc.sh

dev-pred-server:
	./scripts/dev_prediction_server.sh

test-pred-server:
	./scripts/test_prediction_server.sh

dev-trading-client:
	./scripts/dev_trading_client.sh

build-trading-client-container:
	docker build -t jneuv/trading_client:latest -f ./deploy/Dockerfile.trading_client \
  --build-arg PREDICTION_SERVICE_API_KEY=$(PREDICTION_SERVICE_API_KEY) \
  --build-arg API_KEY=${API_KEY} \
  --build-arg API_SECRET=${API_SECRET} .

