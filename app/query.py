"""
Handles querying the vectorstore and sending results to the LLM.
"""

from langchain_community.vectorstores import FAISS
from .llm import generate_response


def query_knowledgebase(
    query: str, persist_path: str = "vectorstore", model: str = "llama3"
) -> str:
    vectorstore = FAISS.load_local(
        persist_path, embeddings=None, allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = (
        f"Answer the following question based on the context:\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    return generate_response(prompt, model=model)
