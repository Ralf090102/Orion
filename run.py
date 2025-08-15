"""
Simple interactive runner for Orion ingestion and querying.
"""

import sys
from app.ingest import build_vectorstore
from app.query import query_knowledgebase
from app.config import Config
from app.utils import log_info, log_success, log_error, validate_path


def main():
    mode = ""
    while mode not in {"ingest", "query"}:
        mode = input("Choose mode: (ingest/query): ").strip().lower()
        if mode not in {"ingest", "query"}:
            log_error("Invalid choice. Please enter 'ingest' or 'query'.")

    if mode == "ingest":
        folder = input("Enter path to documents folder: ").strip()
        try:
            validate_path(folder, must_exist=True)
            result = build_vectorstore(
                folder, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
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
        question = input("Enter your question: ").strip()
        model = (
            input(f"Model to use [{Config.LLM_MODEL}]: ").strip() or Config.LLM_MODEL
        )
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
            sys.exit(1)


if __name__ == "__main__":
    main()
