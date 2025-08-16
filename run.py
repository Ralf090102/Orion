"""
Simple interactive runner for Orion ingestion and querying.
"""

import sys
from app.ingest import rebuild_vectorstore, incremental_vectorstore
from app.query import query_knowledgebase
from app.config import Config
from app.utils import log_info, log_success, log_error, validate_path


def main():
    # --- Pre-flight checks ---
    import requests
    import importlib.util

    def check_ollama_connection(base_url):
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def check_model_pulled(base_url, model_name):
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                return any(m.get("name", "") == model_name for m in tags)
        except Exception:
            pass
        return False

    def pull_model(base_url, model_name):
        try:
            resp = requests.post(
                f"{base_url}/api/pull", json={"name": model_name}, timeout=60
            )
            return resp.status_code == 200
        except Exception:
            return False

    def check_python_package(pkg_name):
        return importlib.util.find_spec(pkg_name) is not None

    config = Config.from_env()
    ollama_url = config.ollama_base_url
    required_model = config.embedding_model

    # Check Ollama connection
    if not check_ollama_connection(ollama_url):
        log_error(f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?")
        sys.exit(1)

    # Check if embedding model is pulled
    if not check_model_pulled(ollama_url, required_model):
        log_info(f"Model '{required_model}' not found. Attempting to pull...")
        if pull_model(ollama_url, required_model):
            log_success(f"Model '{required_model}' pulled successfully.")
        else:
            log_error(
                f"Failed to pull model '{required_model}'. Please pull manually: ollama pull {required_model}"
            )
            sys.exit(1)

    # Check required Python packages
    missing_pkgs = []
    for pkg in ["docx2txt", "langchain_ollama"]:
        if not check_python_package(pkg):
            missing_pkgs.append(pkg)
    if missing_pkgs:
        log_error(
            f"Missing required Python packages: {', '.join(missing_pkgs)}. Please install them via pip."
        )
        sys.exit(1)

    mode = ""
    while mode not in {"ingest", "query"}:
        mode = input("Choose mode: (ingest/query): ").strip().lower()
        if mode not in {"ingest", "query"}:
            log_error("Invalid choice. Please enter 'ingest' or 'query'.")

    if mode == "ingest":
        folder = input("Enter path to documents folder: ").strip()
        ingest_mode = ""
        while ingest_mode not in {"rebuild", "increment"}:
            ingest_mode = input("Ingest mode: (rebuild/increment): ").strip().lower()
            if ingest_mode not in {"rebuild", "increment"}:
                log_error("Invalid choice. Please enter 'rebuild' or 'increment'.")

        config = Config.from_env()
        try:
            validate_path(folder, must_exist=True)

            if ingest_mode == "rebuild":
                result = rebuild_vectorstore(
                    folder,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    embedding_model=config.embedding_model,
                )
            else:  # increment
                result = incremental_vectorstore(
                    folder,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    embedding_model=config.embedding_model,
                )

            if result:
                log_success("== Ingestion complete! ==")
            else:
                log_error("== Ingestion failed! ==")
                sys.exit(1)

        except Exception as e:
            log_error(f"Ingestion error: {e}")
            sys.exit(1)

    elif mode == "query":
        config = Config.from_env()
        model = (
            input(f"Model to use [{config.llm_model}]: ").strip() or config.llm_model
        )
        while True:
            question = input("Enter your question (or type 'exit' to quit): ").strip()
            if question.lower() == "exit":
                log_info("Exiting query mode.")
                break
            try:
                result = query_knowledgebase(question, model=model)
                answer = result["answer"] if isinstance(result, dict) else result
                log_success("\n== Answer ==")
                print(answer)
                if isinstance(result, dict) and result.get("sources"):
                    log_info("Sources:")
                    for src in result["sources"]:
                        print(f" - {src}")
            except Exception as e:
                log_error(f"Query error: {e}")


if __name__ == "__main__":
    main()
