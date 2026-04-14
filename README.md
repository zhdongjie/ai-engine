# ai-engine

ai-engine is a lightweight AI execution engine for running LLM chains, agents, and workflows as APIs.

It exposes AI capabilities as microservices, allowing applications to integrate LLM features through standard HTTP APIs.
The engine focuses on AI inference and workflow execution, leaving business logic, authentication, and orchestration to
upstream services.

---

# Key Features

- Run LLM chains as APIs
- Build agent-based workflows
- Support RAG pipelines
- Stream responses via HTTP
- Multi-provider LLM support

---

# Architecture

The system follows a two-layer architecture.

```
Client / App
    |
   HTTP
    v
+------------------+
|   AI Gateway     |
|  (Business API)  |
+---------+--------+
          |
          v
+------------------+
|    ai-engine     |
|  Chains/Agents   |
|  Workflows/RAG   |
+---------+--------+
          |
          v
+------------------+
|   LLM Providers  |
| OpenAI / Zhipu   |
| Claude / Others  |
+------------------+
```

## Responsibility Separation

| Component         | Responsibility                               |
|-------------------|----------------------------------------------|
| ai-engine         | AI inference and workflow execution          |
| Gateway / BFF     | Authentication, business APIs, orchestration |

This separation keeps the AI layer stateless, scalable, and reusable.

---

# Tech Stack

## Language

Python 3.11+

## AI Frameworks

- LangChain
- LangGraph
- LangServe

LangServe automatically exposes LangChain Runnable / Chains / Agents as REST APIs.

## Web Framework

- FastAPI
- Uvicorn

## Storage

- PostgreSQL
- pgvector
- Redis

## Observability

- LangSmith

## DevOps

- Poetry
- Docker
- Docker Compose

---

# Project Structure

```
ai-engine/
  src/ai_engine/
    api/
    chains/
    core/
    graphs/
    infra/
    knowledge/
    models/
    repository/
    schemas/
    utils/
    server.py
  tests/
  resource/
  scripts/
  main.py
  pyproject.toml
  README.md
```

---

# Quick Start

## 1 Install dependencies

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

## 2 Configure environment

Create `.env` (or `.env.dev`):

```
ENV=dev
QWEN_API_KEY=your-key
```

## 3 Start the server

```bash
poetry run python main.py
```

Server runs at `http://127.0.0.1:8000` by default (configurable via `PROJECT_HOST` / `PROJECT_PORT`).

---

# Example API

Invoke a chain:

```
POST /v1/chat/invoke
```

Example request:

```bash
curl http://127.0.0.1:8000/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "hello"}'
```

## Streaming

```
POST /v1/chat/stream
```

## Batch

```
POST /v1/chat/batch
```

---

# Example Server

```python
from fastapi import FastAPI
from langserve import add_routes
from ai_engine.chains.chat_chain import chat_chain

app = FastAPI()

add_routes(
    app,
    chat_chain,
    path="/v1/chat",
)
```

This automatically generates:

```
POST /v1/chat/invoke
POST /v1/chat/stream
POST /v1/chat/batch
POST /v1/chat/playground
```

---

# Roadmap

- Chat Chain
- RAG Engine
- Agent Workflows
- Tool Integration
- Memory System
- Multi-model support
- Observability integration

---

# Docker

```bash
docker compose up --build
```

---

# License

MIT License

---

# Philosophy

ai-engine focuses only on AI execution:

- Business APIs -> Gateway
- AI inference -> ai-engine

This architecture keeps AI services modular, scalable, and reusable.
