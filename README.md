# ai-engine

**ai-engine** is a lightweight **AI execution engine** for running **LLM Chains, Agents, and Workflows as APIs**.

It is designed to expose **AI capabilities as microservices**, allowing applications to integrate LLM features through
standard HTTP APIs.

The engine focuses only on **AI inference and workflow execution**, leaving business logic, authentication, and
orchestration to upstream services.

---

# ✨ Key Features

* Run **LLM Chains** as APIs
* Build **Agent-based workflows**
* Support **RAG pipelines**
* Stream responses via HTTP
* Deploy AI capabilities as **microservices**
* Multi-provider LLM support

---

# 🧠 Architecture

The system follows a **two-layer architecture**.

```
Client / App
      │
      │ HTTP
      ▼
+------------------+
|   AI Gateway     |
|  (Business API)  |
+---------+--------+
          │
          │
          ▼
+------------------+
|    ai-engine     |
|                  |
|  Prompt Engine   |
|  Chains          |
|  Agents          |
|  Workflows       |
|  RAG Pipelines   |
+---------+--------+
          │
          │
          ▼
+------------------+
|   LLM Providers  |
| OpenAI / Zhipu   |
| Claude / Others  |
+------------------+
```

### Responsibility Separation

| Component         | Responsibility                               |
|-------------------|----------------------------------------------|
| **ai-engine**     | AI inference and workflow execution          |
| **Gateway / BFF** | Authentication, business APIs, orchestration |

This separation keeps the AI layer **stateless, scalable, and reusable**.

---

# 🛠 Tech Stack

### Language

Python **3.11+**

---

### AI Frameworks

* LangChain
* LangGraph
* LangServe

LangServe automatically exposes **LangChain Runnable / Chains / Agents as REST APIs**.

---

### Web Framework

* FastAPI
* Uvicorn

---

### Storage

* PostgreSQL
* pgvector
* Redis

---

### Observability

* LangSmith

---

### DevOps

* Poetry
* Docker
* Docker Compose

---

# 📦 Project Structure

```
ai-engine
│
├── app
│   │
│   ├── chains          # LLM chains
│   │
│   ├── agents          # Agent implementations
│   │
│   ├── workflows       # LangGraph workflows
│   │
│   ├── prompts         # Prompt templates
│   │
│   ├── llm             # LLM provider abstraction
│   │
│   ├── infra           # Vector store / retriever
│   │
│   ├── server.py       # FastAPI entrypoint
│   │
│   └── config.py       # Configuration
│
├── tests
│
├── docker
│
├── pyproject.toml
│
└── README.md
```

---

# 🚀 Quick Start

### 1 Install dependencies

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

---

### 2 Configure environment

Create `.env`

```
OPENAI_API_KEY=your-key
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o
```

---

### 3 Start the server

```
poetry run python app/server.py
```

Server runs at:

```
http://localhost:8000
```

---

# 🔌 Example API

Invoke a chain:

```
POST /chat/invoke
```

Example request:

```
curl http://localhost:8000/chat/invoke \
-H "Content-Type: application/json" \
-d '{"input": "hello"}'
```

---

### Streaming

```
POST /chat/stream
```

---

### Batch

```
POST /chat/batch
```

---

# 🧩 Example Server

```
from fastapi import FastAPI
from langserve import add_routes
from app.chains.chat_chain import chat_chain

app = FastAPI()

add_routes(
    app,
    chat_chain,
    path="/chat"
)
```

This automatically generates:

```
POST /chat/invoke
POST /chat/stream
POST /chat/batch
POST /chat/playground
```

---

# 🗺 Roadmap

* Chat Chain
* RAG Engine
* Agent Workflows
* Tool Integration
* Memory System
* Multimodel support
* Observability integration

---

# 🐳 Docker (coming soon)

The project will support containerized deployment using Docker.

---

# 📜 License

MIT License

---

# 💡 Philosophy

**ai-engine focuses only on AI execution.**

* Business APIs → Gateway
* AI inference → ai-engine

This architecture keeps AI services:

* modular
* scalable
* reusable
