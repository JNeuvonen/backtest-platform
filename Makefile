.PHONY: build-pred-service-container

build-pred-service-container:
	docker buildx build --platform linux/amd64 -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=$(DATABASE_URI) . --load


build-trading-client-container:
	docker buildx build --platform linux/amd64 -t jneuv/trading_client:latest -f ./deploy/Dockerfile.trading_client \
  --build-arg PREDICTION_SERVICE_API_KEY=$(PREDICTION_SERVICE_API_KEY) \
  --build-arg API_KEY=${API_KEY} \
  --build-arg API_SECRET=${API_SECRET} . --load

post-test-strategy:
	python scripts/pred_server_post_strategy.py

dev-platform:
	./scripts/dev_platform.sh

test-platform:
	./scripts/test_platform.sh

loc:
	./scripts/loc.sh


dev-analytics:
	./scripts/dev_analyticsweb.sh

dev-pred-server:
	./scripts/dev_prediction_server.sh

dev-pred-serv:
	./scripts/dev_prediction_server.sh

dev-pred-service:
	./scripts/dev_prediction_server.sh

pred-service-gen-migr-file:
	./scripts/alembic_autogen_pred_service.sh -d $(DATABASE_URI) -m $(MIGRATION_MESSAGE)

pred-service-run-migr-file:
	./scripts/alembic_migrate_pred_service.sh $(DATABASE_URI)


close-all-positions:
	./scripts/close_all_positions.sh -k $(API_KEY) -s $(API_SECRET)

test-pred-server:
	./scripts/test_prediction_server.sh

dev-trading-client:
	./scripts/dev_trading_client.sh

cleanup-bot-msgs-slack:
	cd scripts && cd slack_automation && node index
