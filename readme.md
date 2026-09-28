# RAG Chatbot

Production-ready Retrieval-Augmented Generation (RAG) chatbot for querying any domain documents. The project includes document ingestion, Milvus vector retrieval, Gemini-based answer generation, multilingual support, persistent multi-session chat memory, Redis semantic caching, and a ChatGPT-style React frontend integrated with the FastAPI backend.

## Key Features

- End-to-end RAG pipeline for domain-specific document question answering.
- Document ingestion and cleaning pipeline for PDF/markdown style knowledge sources.
- Google Gemini embeddings and Gemini response generation.
- Milvus vector database for semantic retrieval.
- Google Discovery Engine reranking for improving retrieved context quality.
- SQLite persistent memory for multi-session chat history.
- Redis semantic cache for repeated-query acceleration.
- Multilingual query handling with Google Cloud Translation.
- DeepEval 3-Level Evaluation Pipeline for component, pipeline, and application evaluation, safety benchmarking, and automated regression testing.
- Source metadata and signed GCS document links in answers.
- React frontend with New Chat, session switching, saved chat history, normal mode, and Deepthink mode.
- Docker and Docker Compose setup for backend, frontend, Redis, Milvus, etcd, and MinIO.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Frontend | React, TypeScript, Vite, Nginx |
| Backend API | FastAPI, Pydantic, SQLAlchemy, Gunicorn, Uvicorn |
| LLM and Embeddings | Google Gemini, LangChain Google GenAI |
| Vector Database | Milvus |
| Reranking | Google Discovery Engine Rank API |
| Translation | Google Cloud Translation |
| Storage Links | Google Cloud Storage signed URLs |
| Persistent Memory | SQLite, sqlite-vec |
| Semantic Cache | Redis Stack / RediSearch vector index |
| Evaluation & Testing | DeepEval, PyTest |
| Deployment | Docker, Docker Compose |

## High-Level Architecture

```text
User
  |
  v
React Frontend (localhost:3000)
  |
  v
FastAPI Backend (localhost:5000)
  |
  +--> Session APIs -> SQLite chat_messages / chat_sessions
  |
  +--> Query API
        |
        +--> Language detection and translation
        |
        +--> Security & Safety Guard (PII Masking, Prompt Injection/Jailbreak Check)
        |
        +--> Gemini embedding generation
        |
        +--> Redis semantic cache lookup
        |
        +--> SQLite semantic memory lookup
        |
        +--> Milvus vector retrieval
        |
        +--> Google Discovery Engine reranking
        |
        +--> Gemini answer generation
        |
        +--> DeepEval Bridge Evaluation (Precision, Recall, Relevance, Faithfulness, Safety)
        |
        +--> GCS signed source URLs
        |
        +--> Save response to SQLite memory and chat history
```

## Project Structure

```text
.
|-- main.py                              # FastAPI app and chat/query APIs
|-- Dockerfile                           # Backend production image
|-- docker-compose.yml                   # Full stack deployment
|-- requirements.txt                     # Python dependencies
|-- config/
|   `-- config.yaml                      # LLM and embedding model config
|-- evaluation_pipeline/                 # DeepEval 3-Level Evaluation Pipeline
|   |-- config.py                        # Model settings & evaluation thresholds
|   |-- deepeval_wrapper.py              # DeepEval LLMTestCase & metric adapters
|   |-- dataset_generator.py             # Golden dataset manager & synthetic QnA generator
|   |-- datasets/                        # Golden & safety datasets
|   |   |-- golden_dataset.json
|   |   `-- safety_test_suite.json
|   |-- component_eval.py                # Level (a): Retriever & Generator isolation evals
|   |-- pipeline_eval.py                 # Level (b): End-to-End RAG pipeline evals
|   |-- app_eval.py                      # Level (c): App quality, safety & operational evals
|   `-- runner.py                        # Regression testing CLI orchestrator & reports
|-- src/
|   |-- step_1_chunking.py               # Chunking utilities
|   |-- step_2_embedding.py              # Embedding logic
|   |-- step_3_llm_loaders.py            # Main RAG orchestration
|   |-- step_4_processing.py             # Retrieval, rerank, Gemini response
|   |-- step_5_prompt.py                 # Prompt template
|   |-- step_6_reranker.py               # Google Discovery Engine reranker
|   |-- step_7_utility.py                # Output utility helpers
|   |-- step_8_session_history.py        # Recent session history loader
|   `-- pipeline/                        # RAG Enterprise Guards & Evaluation Bridge
|       |-- security.py                  # PII Masking & Prompt Injection Detection
|       |-- evaluation.py                # Bridge connecting main app to evaluation_pipeline
|       |-- validator.py                 # Structural & Output Schema Validation
|       |-- circuit_breaker.py           # Resiliency & Fallback Circuit Breaker
|       |-- cost_governance.py           # Cost Tracking & Token Usage Manager
|       |-- rate_limiter.py              # API Request Rate Limiter
|       `-- monitoring.py                # Telemetry & LangSmith Tracer
|-- milvus_database/
|   |-- config.py                        # Milvus collection and connection config
|   |-- factory_client.py                # Milvus client factory
|   |-- milvus_db.py                     # Insert/search helpers
|   `-- milvus_loading.py                # Milvus loading/bootstrap
|-- caching_history/
|   |-- caching/redis_semantic_cache.py  # Redis vector cache
|   `-- chat_history/                    # SQLAlchemy chat session/message DB
|-- persistant_memory/
|   |-- loading_and_saving_chat.py       # SQLite persistent semantic memory
|   `-- load_chat_history.py             # Redis/cache retrieval helpers
|-- multilingual_pipeline/
|   |-- language_detection.py            # Language detection
|   `-- conversion.py                    # Google Translation integration
|-- url_integration/
|   `-- gcs_url.py                       # Google Cloud Storage signed URL generation
|-- data_ingestion_and_cleaning_03/      # OCR/PDF cleaning pipeline
|-- frontend/
|   |-- src/                             # React application source
|   |-- Dockerfile                       # Frontend production image
|   `-- nginx.conf                       # Nginx SPA + API proxy config
`-- chat_history.db                      # SQLite database for local chat memory
```

## Backend APIs

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Container health check |
| `POST` | `/chat/session` | Create a new chat session |
| `GET` | `/sessions` | List saved sessions |
| `GET` | `/sessions/{session_id}/chats` | Load all messages for one session |
| `POST` | `/chat/message` | Store a manual chat message |
| `POST` | `/query` | Ask a RAG question using a session id |

Example `/query` payload:

```json
{
  "question": "Explain the IRS and LRS classification process.",
  "answer_type": "deepthink",
  "session_id": "existing-session-id"
}
```

## RAG Workflow

1. User creates or opens a chat session from the React frontend.
2. Frontend sends the question, answer mode, and `session_id` to `/query`.
3. Backend detects language and translates non-English queries to English.
4. Security guard sanitizes input and checks for prompt injection or jailbreak patterns.
5. Gemini embedding model converts the query into a vector.
6. Redis semantic cache checks for a high-similarity previous answer.
7. SQLite semantic memory checks persisted historical answers.
8. If no cache hit is found, Milvus retrieves relevant document chunks.
9. Google Discovery Engine reranks retrieved chunks.
10. Gemini generates a structured JSON answer using the retrieved context and recent session history.
11. DeepEval evaluation bridge evaluates the query, answer, and context across precision, recall, relevance, faithfulness, and safety.
12. Backend enriches the response with metadata and signed GCS source URLs.
13. User and assistant messages are saved to SQLite chat history.
14. High-value answers are cached in Redis for faster future responses.

## Evaluation Pipeline & Regression Testing

The evaluation framework is located in `evaluation_pipeline/` outside the `src` folder. It uses **DeepEval** to assess the application across 3 distinct levels:

### Level (a): Component-Level Evaluation (`component_eval.py`)
- **Retriever Component**: Evaluated independently on `precision@k` and `recall@k` against ground truth context chunks.
- **Generator Component**: Evaluated in isolation on custom QnA golden datasets for:
  - **Faithfulness / Hallucination Detection**
  - **Answer Relevance**
  - **Citation / Reference Alignment**
  - **Reference-Free LLM Judge** (`GEval`)

### Level (b): Pipeline-Level Evaluation (`pipeline_eval.py`)
- Evaluates the end-to-end RAG response using mandatory metrics:
  - **Answer Relevance** (`AnswerRelevancyMetric`)
  - **Context Relevance** (`ContextualRelevancyMetric` / `ContextualPrecisionMetric`)
  - **Faithfulness** (`FaithfulnessMetric`)

### Level (c): Application-Level Evaluation (`app_eval.py`)
- Evaluates full application performance across 5 dimensions:
  - **Correctness**: Factual alignment (`GEval`)
  - **Completeness**: Response detail depth (`GEval`)
  - **Style**: Tone, grammar, and structural formatting (`GEval`)
  - **Safety Evals**:
    - **Toxicity**: DeepEval `ToxicityMetric`
    - **PII Injection**: Automated masking & leakage detection via `security_guard`
    - **Jailbreaks**: Prompt injection heuristic pattern detection
  - **Operational Evals**: Latency (ms), Cost (USD), and Token consumption

### Running Regression Testing

To run the complete regression test suite across golden datasets and generate a detailed report:

```bash
python -m evaluation_pipeline.runner
```

Report files are saved in `evaluation_pipeline/reports/eval_report_<timestamp>.json`.

## Memory and Caching

### SQLite Chat History

The app stores multi-session conversations in SQLite using:

- `chat_sessions`
- `chat_messages`

This powers the frontend session list and reloadable chat history.

### SQLite Semantic Memory

The persistent memory layer stores:

- original user question
- answer JSON
- query vector
- confidence score
- hit count
- session id

This allows semantic reuse of older answers across sessions.

### Redis Semantic Cache

Redis Stack is used with a RediSearch vector index. It stores frequently used RAG answers and performs cosine similarity lookup before running the full RAG pipeline.

Expected benefit: repeated or near-duplicate questions can avoid full Milvus + reranker + LLM execution, reducing repeated-query latency by up to 60-70% depending on cache hit rate and model latency.

## Frontend Workflow

The frontend is located in `frontend/`.

Main capabilities:

- Create a new chat session.
- Switch between existing sessions.
- Load saved messages from SQLite-backed APIs.
- Ask questions in normal or Deepthink mode.
- Display assistant answers, table output, follow-up questions, and source links.
- Use Nginx in production to serve the SPA and proxy `/api/*` requests to FastAPI.

Local frontend development:

```bash
cd frontend
pnpm install
pnpm dev
```

Production frontend build:

```bash
cd frontend
pnpm build
```

## Environment Variables

Create a `.env` file in the project root. Do not commit real secrets.

```env
GOOGLE_API_KEY=your-google-api-key
MILVUS_HOST=standalone
MILVUS_PORT=19530
DEEPEVAL_LLM_MODEL=gpt-4o-mini
```

Docker Compose also sets:

```env
REDIS_HOST=redis-stack
REDIS_PORT=6379
GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
DATABASE_URL=sqlite:////app/data/chat_history.db
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

The Google service account JSON must be available as:

```text
service-account.json
```

It is mounted into the backend container at:

```text
/app/service-account.json
```

## Docker Deployment

Build and run the full stack:

```bash
docker compose up --build
```

Services:

| Service | Port | Purpose |
| --- | --- | --- |
| `frontend` | `3000` | React UI served by Nginx |
| `backend` | `5000` | FastAPI RAG API |
| `redis` | `6379`, `8001` | Redis Stack and RedisInsight |
| `standalone` | `19530` | Milvus standalone |
| `minio` | `9005`, `9006` | Milvus object storage |
| `etcd` | internal | Milvus metadata store |

Open the application:

```text
http://localhost:3000
```

Backend health check:

```text
http://localhost:5000/health
```

## Data Ingestion Workflow

1. Raw documents are cleaned using the PDF/OCR cleaning pipeline.
2. Text is chunked into smaller semantic units.
3. Gemini embeddings are generated for each chunk.
4. Chunks and metadata are inserted into Milvus.
5. Runtime queries retrieve the most relevant chunks from Milvus.

Typical ingestion components:

- `data_ingestion_and_cleaning_03/pdf_cleaning.py`
- `data_ingestion_and_cleaning_03/ocr_pipeline/`
- `src/step_1_chunking.py`
- `src/step_2_embedding.py`
- `milvus_database/milvus_db.py`
- `workflow_milvus.py`

## Deepthink Mode

Deepthink mode bypasses quick cache-first answering and runs the full retrieval pipeline:

1. Query embedding
2. Milvus retrieval
3. Google reranking
4. Context-aware Gemini response generation
5. Persistent memory save

This mode is useful for complex domain questions that require deeper document grounding.

## Production Notes

- Keep `.env`, `service-account.json`, and `chat_history.db` out of version control.
- Use `DATABASE_URL` to control SQLite location in Docker and local runs.
- Redis cache is optional for correctness but improves latency for repeated queries.
- Milvus must contain the target collection and partition before production use.
- Google Cloud APIs required by the project should be enabled for the service account:
  - Gemini / Generative AI
  - Cloud Translation
  - Discovery Engine ranking
  - Cloud Storage



