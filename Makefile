.PHONY: build-pred-service-container

build-pred-service-container:
	docker buildx build --platform linux/amd64 -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=$(DATABASE_URI) . --load


post-test-strategy:
	python scripts/pred_server_post_strategy.py
