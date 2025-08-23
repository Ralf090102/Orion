#!/usr/bin/env python3
"""
Test individual imports from run.py to identify problematic modules
"""


def test_import(module_name, import_statement):
    """Test a single import and report results"""
    print(f"Testing: {import_statement}")
    try:
        exec(import_statement)
        print(f"✅ {module_name}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {module_name}: ERROR - {e}")
        return False


def main():
    """Test all imports from run.py individually"""
    print("🧪 Testing run.py imports individually...\n")

    imports_to_test = [
        (
            "core.rag.ingest",
            "from core.rag.ingest import rebuild_vectorstore, incremental_vectorstore, "
            "rebuild_vectorstore_async, incremental_vectorstore_async",
        ),
        (
            "core.rag.query",
            "from core.rag.query import query_knowledgebase, query_with_performance_optimizations",
        ),
        ("core.utils.config", "from core.utils.config import Config"),
        (
            "core.utils.orion_utils",
            "from core.utils.orion_utils import log_info, log_success, log_error, log_warning, set_verbose_mode",
        ),
        (
            "core.rag.llm",
            "from core.rag.llm import get_available_models, check_ollama_connection",
        ),
        (
            "core.utils.caching",
            "from core.utils.caching import get_global_cache_stats, clear_global_cache",
        ),
    ]

    success_count = 0
    total_count = len(imports_to_test)

    for module_name, import_statement in imports_to_test:
        if test_import(module_name, import_statement):
            success_count += 1
        print()

    print(f"📊 Results: {success_count}/{total_count} imports successful")

    if success_count == total_count:
        print("🎉 All imports working!")
    else:
        print("⚠️ Some imports failed - need to fix these first")


if __name__ == "__main__":
    main()
