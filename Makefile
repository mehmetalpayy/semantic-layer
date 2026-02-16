.PHONY: run-main run-build-mdl run-retrieval docker-up docker-down docker-logs

run-main:
	uv run python main.py

run-build-mdl:
	uv run python -m semantic_layer.build_mdl

run-retrieval:
	uv run python -m semantic_layer.pgvector_retrieval

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=200
