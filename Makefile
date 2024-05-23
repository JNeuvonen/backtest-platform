.PHONY: build-pred-service-container

#CONTAINERS

build-pred-service-container:
	docker buildx build --platform linux/amd64 -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=$(DATABASE_URI) . --load


build-pred-service-container-local:
	docker build -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=$(DATABASE_URI) . --load


build-trading-client-container:
	docker buildx build --platform linux/amd64 -t jneuv/trading_client:latest -f ./deploy/Dockerfile.trading_client \
  --build-arg PREDICTION_SERVICE_API_KEY=$(PREDICTION_SERVICE_API_KEY) \
  --build-arg API_KEY=${API_KEY} \
  --build-arg API_SECRET=${API_SECRET} . --load


#DEV STARTUP SCRIPTS

dev-platform:
	./scripts/dev_platform.sh

dev-trading-client:
	./scripts/dev_trading_client.sh


dev-analytics-www:
	./scripts/dev_analyticsweb.sh

dev-analytics-serv:
	./scripts/dev_analytics_server.sh

dev-pred-serv:
	./scripts/dev_prediction_server.sh

#TESTS

test-platform:
	./scripts/test_platform.sh


test-pred-server:
	./scripts/test_prediction_server.sh

test-analytics-server:
	./scripts/test_analytics_server.sh



#MIGRATIONS

pred-service-gen-migr-file:
	./scripts/alembic_autogen_pred_service.sh -d $(DATABASE_URI) -m $(MIGRATION_MESSAGE)

pred-service-run-migr-file:
	./scripts/alembic_migrate_pred_service.sh $(DATABASE_URI)

#UTILITY

close-all-positions:
	./scripts/close_all_positions.sh -k $(API_KEY) -s $(API_SECRET)


post-test-strategy:
	python scripts/pred_server_post_strategy.py

cleanup-bot-msgs-slack:
	cd scripts && cd slack_automation && node index


loc:
	./scripts/loc.sh
