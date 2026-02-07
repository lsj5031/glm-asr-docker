.PHONY: build run up shell logs restart clean help

help:
	@echo "GLM-ASR Docker Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       Build the Docker image"
	@echo "  run         Run the container with GPU support"
	@echo "  up          Build and run (production best practice)"
	@echo "  shell       Open a shell in a running container"
	@echo "  logs        Show container logs"
	@echo "  restart     Restart the containers"
	@echo "  clean       Stop and remove the container"
	@echo "  help        Show this help message"

build:
	docker build -t glm-asr .

run:
	docker compose up -d

up:
	docker compose up -d --build --remove-orphans

shell:
	docker exec -it glm-asr /bin/bash

logs:
	docker compose logs -f

restart:
	docker compose restart

clean:
	docker compose down
	docker rmi glm-asr || true
