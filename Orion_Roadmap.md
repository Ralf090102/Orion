# Orion Roadmap — Local RAG Assistant

This document is a detailed, step-by-step engineering roadmap for "Orion" — a purely local RAG (Retrieval-Augmented Generation) assistant that uses one or more folder paths as its knowledge base. The roadmap maps to the current `config.py` for parameter names and defaults, highlights milestones, testing strategies, and implementation notes.

---

## Goals and constraints
- Fully local: no external cloud LLM/compute required. Use Ollama for LLM runtime locally.
- Multiple folder knowledge base: users can point `Orion` at one or more folders. The system will monitor them (watchdog) and ingest updates incrementally.
- Modular architecture: clear separation between retrieval, reranking, generation, backend, frontend, and system integration.
- Optional GPU acceleration and optional heavy extras (OCR, media) via extras in packaging.
- UX: fast interactive queries, configurable latency/quality trade-offs via `config.py`.

---

## How to read this roadmap
- Each section lists: purpose, concrete tasks (with file-level mapping), priority, tests, and milestones.
- Implementation notes reference `config.py` keys where appropriate and suggest sensible defaults.

---

## I. Retrieval Phase (High priority)
Purpose: Ingest files, create embeddings, store vectors, search, and produce candidate passages. This is the most critical and largest part of Orion.

Core files (existing in repo):
- `retrieval/vector_space.py` (indexing, storage adapter)
- `retrieval/embeddings.py` (model wrappers, batching)
- `retrieval/search.py` (vector + hybrid search orchestration)
- `retrieval/reranker.py` (cross-encoder reranking)
- `retrieval/retrieval.py` (high-level pipeline that orchestrates ingest -> search -> rerank)
- `retrieval/watchdog.py` (file watcher integration for incremental ingestion)

High-level checklist and step-by-step tasks:

1) vector_space.py — Vector store abstraction & utilities
- Purpose: Provide a thin abstraction over different vector stores (Chroma, FAISS, Annoy). Support persist, snapshot, and concurrency-safe writes.
- Tasks:
  - Define VectorStore interface: upsert(docs: list[Document]), query(embeddings, top_k), persist(), load(), delete_collection().
  - Implement Chroma adapter (default) using `chromadb` and `config.rag.vectorstore` settings such as `persist_directory`, `collection_name`, `distance_metric`, `use_gpu`, `batch_size`.
  - Add FAISS adapter (optional) behind same interface for users who prefer it.
  - Implement efficient bulk-upsert with configurable `batch_size` (use `config.rag.vectorstore.batch_size`).
  - Add safe persist_immediately behavior controlled by `config.rag.vectorstore.persist_immediately`.
- Tests / Milestones:
  - Unit test: round-trip upsert -> query returns inserted doc id.
  - Performance test: bulk ingest 10k chunks with `batch_size` param.

2) embeddings.py — Model wrapper and caching
- Purpose: Wrap sentence-transformers (and optionally other embedding backends) and provide batched embedding with caching.
- Tasks:
  - Implement EmbeddingClient that honors `config.rag.embedding.model`, `batch_size`, and `timeout`.
  - Add file-level caching strategy controlled by `config.rag.embedding.cache_embeddings` (persist embeddings per-document via a checksum file or DB).
  - Implement fallback to CPU if GPU disabled; support optional quantized/onnx flows for CPU heavy models.
  - Expose `embed_documents(docs: List[str]) -> np.ndarray` and `embed_query(q: str)`.
- Tests:
  - Unit test: embedding of sample texts returns consistent vector dims.
  - Integration: embedding -> index -> search returns high-similarity for identical text.

3) search.py — Hybrid retrieval orchestration
- Purpose: Merge vector search and BM25/keyword search results and apply MMR if enabled.
- Tasks:
  - Implement text index or use `rank_bm25` for keyword search.
  - Provide `hybrid_search(query, top_k, config)` that:
    - Embeds query
    - Vector query top `mmr_fetch_k` (config.rag.retrieval.mmr_fetch_k or larger)
    - BM25 top `mmr_fetch_k`
    - Merge sets, apply hybrid weighting (`semantic_weight`, `keyword_weight`), deduplicate
    - If `enable_mmr` is True: run MMR selecting top `default_k` with `mmr_diversity_bias`
  - Return candidate passage objects with metadata for reranker
- Tests:
  - Unit: confirm hybrid merge strategy returns union and weights applied.
  - Integration: measured recall vs simple vector search on small dataset.

4) reranker.py — Cross-encoder reranking
- Purpose: Take candidate passages and rerank them with cross-encoder models from `config.rag.reranker.model`.
- Tasks:
  - Wrap sentence-transformers CrossEncoder or transformers pipeline.
  - Batch processing using `config.rag.reranker.batch_size` and `enable_batch_processing`.
  - Provide `rerank(query, candidates) -> ranked_candidates` and `score_threshold` filtering.
  - Cache reranker scores if needed for repeated queries.
- Tests:
  - Unit test: reranker produces higher scores for relevant candidate vs irrelevant.
  - Integration: measure improvements after reranking (precision@k).

5) retrieval.py — Top-level retrieval pipeline
- Purpose: Glue together `embeddings`, `vector_space`, `search`, and `reranker` to return top-N passages.
- Tasks:
  - Implement `retrieve(query, k=None, config=None)` that:
    - Uses embedding client to embed query
    - Uses `search.hybrid_search` to get candidates
    - Optionally reranks (config.rag.retrieval.enable_reranking)
    - Returns top `k` passages with metadata
  - Respect timeouts from `config.rag.embedding.timeout` and `config.rag.reranker.timeout`.
  - Produce structured results for generation phase (text, source path, offsets, score).
- Tests:
  - End-to-end test: ingest a small corpus, query, and assert returned passages include ground-truth.

6) watchdog.py — File watcher & incremental ingestion
- Purpose: Watch configured `paths` for file additions/changes/deletes, debounce events, and kick off ingestion.
- Tasks:
  - Implement a `Watcher` class using `watchdog.observers.Observer` with handlers that support debouncing (`config.watchdog.debounce_seconds`) and ignore patterns.
  - On file add/modify: schedule ingestion operation via a worker pool with max_workers=`config.watchdog.max_workers`.
  - On file delete: remove vectors and metadata from vector store.
  - Provide a simple CLI to start/stop the watcher; integrate with backend REST endpoints to enable/disable.
- Tests:
  - Integration test: create/remove files in temp dir and assert ingestion/upsert/delete events happen.

Priority & timeline (suggested)
- Week 1: Implement `embeddings.py`, `vector_space.py` (Chroma adapter), basic `search.py` vector-only.
- Week 2: Add BM25 in `search.py`, implement `retrieval.py` pipeline and small end-to-end test.
- Week 3: Implement `reranker.py` and integrate reranking into retrieval pipeline.
- Week 4: Implement `watchdog.py` with debouncing and incremental ingestion; integrate with pipeline.

---

## II. Generating Phase (High priority)
Purpose: Given retrieved passages, build a prompt for the local LLM (Ollama), call the model, and post-process answers.

Core files to add/extend:
- `generation/prompt_builder.py` — create RAG prompts, add citation formatting, context window management.
- `generation/llm_client.py` — wrapper over `ollama` calls (already partially present in `core/llm.py`); make it robust and add streaming support.
- `generation/generate.py` — orchestrates prompt building, LLM calls, and answer post-processing.

Tasks & details:
1) prompt_builder.py
- Implement chunk selection and concatenation into prompt respecting model context (monitor token count). Use a simple tokenizer (tiktoken-like) or approximate tokens per word if tiktoken not available locally.
- Include citation template: `[source:path:offset]` for each passage.
- Support configurable system prompt from `config.rag.llm.system_prompt` and generation params (temperature/top_p/max_tokens).

2) llm_client.py
- Harden `core/llm.py` behavior: provide sync and stream APIs, explicit exceptions, and retry semantics (configurable on timeouts).
- Expose `generate(prompt, model, options)` and `stream_generate(..., on_token=callable)`.

3) generate.py
- Call retrieval pipeline, build prompt, call llm_client, and return answer with sources.
- Post-process: remove hallucinated citations, apply heuristics to verify short factual answers.

Tests
- Unit: prompt builder output includes citations and stays within token limit.
- Integration: run a sample query against a small knowledge base and assert reasonable answer and presence of sources.

Timeline: 1-2 weeks in parallel with Retrieval Phase.

---

## III. Backend Phase (Medium priority)
Purpose: Expose the retrieval and generation pipeline to the local UI (REST API or IPC), manage ingestion, settings, and provide health checks.

Core files/services:
- `backend/app.py` (FastAPI)
- `backend/api/retrieve.py` (query endpoints)
- `backend/api/ingest.py` (manual ingest endpoints & status)
- `backend/api/watchdog_control.py` (enable/disable watcher endpoints)
- `backend/state.py` (in-memory + optional persistent state for ingestion queue)

Tasks
- Create REST endpoints:
  - POST /query { "q": "..." } -> returns answers + sources
  - POST /ingest { "paths": [...] } -> trigger ingestion on demand
  - GET /status -> returns health: vectorstore reachable, watcher running, models available (use `core/llm.check_ollama_connection`)
- Add authentication (optional for local-only; simple token via env ORION_API_TOKEN).
- Integrate with `set_dependencies.py` to optionally install server extras.

Tests
- API integration tests (use httpx) verifying end-to-end queries.

Timeline: 1-2 weeks, parallel with Retrieval/Generating.

---

## IV. Frontend Phase (Low/Medium priority)
Purpose: Provide a user-friendly UI (Svelte already present in `frontend/`) for querying, ingestion management, and settings.

Files & tasks
- `frontend/src/routes/+page.svelte` — main query UI with streaming answers
- Settings page for model selection, chunking, and watcher paths.
- Progress indicators for ingestion and watcher events.
- Integrate with backend endpoints and show source links (open file in OS using backend helper).

Testing
- E2E tests with Playwright/Vitest for UI flows.

Timeline: 2-3 weeks after core backend stable.

---

## V. System Tray / Tauri Phase (Optional)
Purpose: Provide a lightweight desktop tray app to manage Orion (start/stop watcher, quick query, notifications).

Files & tasks
- `system_tray/` already exists — implement tray script to run background FastAPI server or connect to existing backend.
- Consider a small Tauri wrapper if you want cross-platform native UI (needs separate packaging work).

Timeline: 2-4 weeks after backend & frontend stable.

---

## Cross-cutting concerns
1) Configuration & validation
- Use `config.py` for centralized config. Validate at startup (`get_config(from_env=True)` in CLI/daemon).
- Provide `orion config show` CLI command to print effective config and the model versions.

2) Testing matrix
- Unit tests for each module
- Integration tests for retrieval + rerank + generate (small sample corpus)
- CI: unit-fast on GitHub Actions; integration/manual (label `test-slow`) to avoid CI resource drain.

3) Packaging & deps
- Keep heavy deps (torch, torchvision, OCR) in optional extras (`nvidia-gpu`, `media`) and keep `sentence-transformers` & `transformers` in core.
- `watchdog` added to requirements; consider making it optional if you want truly minimal core.

4) Resource management
- Respect `config.gpu` and `vectorstore.use_gpu` flags.
- Provide `--no-watch` CLI flag for headless ingestion-only mode.

5) Observability & telemetry (local-only)
- Local logging via `utils.setup_logging()` and `config.logging` flags.
- Optional debug endpoint /metrics (Prometheus) guarded by config.

---

## Milestones & deliverables (6–12 weeks roadmap)
- M1 (Week 1–2): Core retrieval (embeddings, vector store, simple search), basic CLI to ingest and query.
- M2 (Week 2–3): Add BM25 hybrid search, MMR, end-to-end retrieval pipeline and basic reranking.
- M3 (Week 3–4): Add watchdog incremental ingestion; end-to-end sample app (FastAPI) and tests.
- M4 (Week 4–6): Prompt builder, LLM integration, streaming answers, frontend query UI (Svelte).
- M5 (Week 6–8): Packaging, optional GPU flows, quantization guidance, documentation, and example datasets.
- M6 (Week 8–12): System tray/Tauri integration, polishing, distribution packaging.

---

## Day-to-day engineering checklist (short)
- Start each day by running a smoke test: small ingest -> query -> rerank -> generate.
- Keep PRs small: 1 feature or fix per PR with unit tests.
- Document any config change in README & `Orion_Roadmap.md`.

---

## Appendix: mapping `config.py` keys to components
- Embedding: `config.rag.embedding.*`
- Chunking: `config.rag.chunking.*`
- Retrieval: `config.rag.retrieval.*` (semantic/keyword weights, MMR)
- Reranker: `config.rag.reranker.*`
- LLM: `config.rag.llm.*` (system_prompt, generation params)
- Vector store: `config.rag.vectorstore.*` (index type, persist, use_gpu)
- Watchdog: `config.watchdog.*`
- GPU/system: `config.gpu.*` and `config.system.*`

---

If you'd like, I can now:
- Create skeleton files for each listed module with TODOs and minimal interfaces (recommended for quick scaffolding).
- Implement the `embeddings.py` + `vector_space.py` skeleton and a small smoke test (ingest 3 small docs and run a query).

Which next action should I take? (I recommend scaffolding the Retrieval modules first.)
