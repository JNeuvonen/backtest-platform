Building from AArch64 to Linux/AMD X86_86: `docker buildx build --platform linux/amd64 -t jneuv/prediction_service:latest -f ./deploy/Dockerfile.prediction_server --build-arg DATABASE_URI=<database_uri_string> . --load`