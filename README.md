<h1 align="center"><strong>Semantic Layer Pipeline with YAML Configs and Secret Settings</strong></h1>

## Overview

Welcome to the **semantic_layer** repository. This project provides a configurable indexing and retrieval pipeline for database schemas and table descriptions used by a semantic-layer database agent. The goal is to supply the agent with the exact schema and table details needed to answer query-related questions. It builds an MDL (Model Definition Language) snapshot from a **remote** PostgreSQL source and stores embeddings and documents in a **local** PostgreSQL (PGVector) store. Configuration is handled via YAML + Pydantic schemas, while secrets are loaded from `.env` using Pydantic Settings.

## What This Project Does

- Connects to a **remote** PostgreSQL database and extracts table metadata.
- Builds and saves an MDL JSON (`semantic_layer/mdl.json`).
- Embeds documents using Azure OpenAI embeddings.
- Stores and retrieves documents using a **local** PGVector-backed PostgreSQL.
- Provides a retrieval workflow with optional column pruning via Azure OpenAI chat.

## Repository Structure

Key files and folders:

1. `main.py`: Entry point for running the full pipeline.
2. `semantic_layer/`: Core pipeline logic.
   - `build_mdl.py`: Builds MDL from the remote database.
   - `pgvector_store.py`: Writes and reads embeddings from local PGVector.
   - `pgvector_retrieval.py`: Retrieval pipeline and optional column pruning.
   - `embedder_litellm.py`: Embedding client wrapper.
3. `configs/`: YAML configs.
   - `default.yaml`: Main configuration file.
4. `config_schemas/`: Pydantic schemas that validate YAML configs.
5. `env.py`: Loads secrets from `.env` (e.g., API keys, DB passwords).
6. `docker-compose.yml`: Local PGVector PostgreSQL.
7. `Makefile`: Common commands (run, docker).

## Installation and Setup

### 1) Create `.env`

Copy `.env.example` to `.env` and fill in the secrets:

- `LOCAL_POSTGRES_PASSWORD`
- `REMOTE_POSTGRES_PASSWORD`
- `AZURE_OPENAI_API_KEY`
- `EMBEDDINGS_AZURE_OPENAI_API_KEY`

### 2) Configure YAML

Edit `configs/default.yaml` for your environment:

- `database.local_db`: Local PGVector connection (host, port, user, db, pool sizes, timeout).
- `database.remote_db`: Remote source connection (host, port, user, db, pool sizes, timeout).
- `infra.azure`: Azure OpenAI chat config (deployment name, endpoint, api version, context window size).
- `summary.long_term`: Azure OpenAI embeddings config (dimension, model, endpoint, deployment, api version).
- `retrieval`: Retrieval knobs (table sizes, score threshold).
- `runtime`: Runtime settings (project_id, retrieval_query, enable_column_pruning, unique_values_concurrency).

### 3) Start Local PostgreSQL (PGVector)

```bash
make docker-up
```

### 4) Build MDL (Remote DB → Local MDL)

```bash
make run-build-mdl
```

### 5) Run the Full Pipeline

```bash
make run-main
```

### Optional: Run Retrieval Only

```bash
make run-retrieval
```

## Configuration Model

The project uses:

- **YAML configs** in `configs/`
- **Pydantic schemas** in `config_schemas/`
- **Secrets** in `.env` via `env.py`

All runtime values are loaded from YAML except secrets, which are strictly read from `.env`.

## Design Decisions

- **Separated DBs**: Remote database is used for metadata extraction, local DB for storage and retrieval.
- **Schema validation**: Pydantic enforces config correctness early.
- **PGVector local store**: Embeddings are indexed locally for fast retrieval.
- **Azure OpenAI integration**: Both embeddings and chat-based column pruning are supported.

## Limitations and Future Work

- **No migrations**: DB schema management is handled by the code (table creation) rather than migrations.
- **Minimal CLI**: Current flow uses `Makefile` targets; a dedicated CLI could improve UX.
- **Config profiles**: Only `default.yaml` is provided. Adding `dev.yaml` / `prod.yaml` is straightforward.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request with a clear description.

## Contact

For feedback or questions, reach out to the maintainer.
