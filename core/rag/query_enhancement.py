"""
Advanced query processing for better retrieval.
"""

from typing import List
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from core.utils.orion_utils import log_info, log_warning, log_debug


class QueryEnhancer:
    """
    Enhances user queries for better retrieval through various techniques.
    """

    def __init__(self, llm_model: str = "mistral"):
        self.llm = ChatOllama(model=llm_model, temperature=0.1)

    def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical document that would answer the query.
        This often retrieves better results than the raw query.
        """
        prompt = f"""Write a detailed paragraph that would perfectly answer this question:
        
Query: {query}

Write as if you're providing the actual answer, not describing what an answer would look like.
Be specific and detailed. Use the terminology and style that would appear in relevant documents."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            log_debug(f"Generated HyDE document: {response.content[:100]}...")
            return response.content.strip()
        except Exception as e:
            log_warning(f"HyDE generation failed: {e}")
            return query

    def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple variations of the query to improve recall.
        """
        prompt = f"""Given this question, generate 3 alternative ways to phrase it that might find relevant documents:

Original: {query}

Generate variations that:
1. Use different terminology/synonyms
2. Approach from different angles  
3. Include related concepts

Format as a simple list:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse response into list
            variations = [
                line.strip().lstrip("123.-• ")
                for line in response.content.split("\n")
                if line.strip()
                and not line.strip().startswith("1.") == line.strip().startswith("2.")
            ]

            # Filter out empty and duplicate variations
            variations = [v for v in variations if v and v != query]
            variations = list(
                dict.fromkeys(variations)
            )  # Remove duplicates while preserving order

            log_debug(f"Generated {len(variations)} query variations")
            return [query] + variations[:3]  # Include original + top 3 variations

        except Exception as e:
            log_warning(f"Query expansion failed: {e}")
            return [query]

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key terms for BM25 search.
        """
        prompt = f"""Extract the most important keywords and phrases from this question for document search:

Question: {query}

List only the key search terms, separated by commas:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            keywords = [kw.strip() for kw in response.content.split(",") if kw.strip()]
            return keywords
        except Exception as e:
            log_warning(f"Keyword extraction failed: {e}")
            # Fallback: simple split
            return query.lower().split()

    def decompose_complex_query(self, query: str) -> List[str]:
        """
        Break down complex multi-part questions into simpler sub-questions.
        This is crucial for handling questions like "What is X and how does it relate to Y?"
        """
        # First, check if the query is complex enough to warrant decomposition
        complexity_indicators = [
            " and ",
            " or ",
            " but ",
            " however ",
            " meanwhile ",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "compare",
            "contrast",
            "relationship",
            "difference",
        ]

        query_lower = query.lower()
        complexity_score = sum(
            1 for indicator in complexity_indicators if indicator in query_lower
        )

        if complexity_score < 2:
            log_info("Query appears simple, skipping decomposition")
            return [query]

        prompt = (
            f"Break down this complex question into 2-4 simpler, focused "
            f"sub-questions that together would fully answer the original question.\n\n"
            f"Original question: {query}\n\n"
            "Sub-questions should:\n"
            "1. Each focus on one specific aspect\n"
            "2. Be answerable independently\n"
            "3. Together cover the full scope of the original question\n"
            "4. Be phrased as complete questions\n\n"
            "Format as a numbered list:"
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse the numbered list
            lines = [
                line.strip() for line in response.content.split("\n") if line.strip()
            ]
            sub_questions = []

            for line in lines:
                # Remove numbering (1., 2., etc.)
                cleaned = line
                if line and line[0].isdigit():
                    # Find the first space after number and dot
                    parts = line.split(".", 1)
                    if len(parts) > 1:
                        cleaned = parts[1].strip()

                if cleaned and len(cleaned) > 10:  # Filter out too-short responses
                    sub_questions.append(cleaned)

            if sub_questions:
                log_info(f"Decomposed query into {len(sub_questions)} sub-questions")
                return sub_questions
            else:
                log_warning("Failed to parse sub-questions, using original")
                return [query]

        except Exception as e:
            log_warning(f"Query decomposition failed: {e}")
            return [query]

    def classify_query_intent(self, query: str) -> str:
        """
        Classify the type of query to optimize search strategy.
        Returns: 'factual', 'analytical', 'procedural', 'comparative', 'general'
        """
        prompt = f"""Classify this question into one of these categories:
- factual: asking for specific facts, definitions, or data
- analytical: asking for analysis, reasoning, or interpretation  
- procedural: asking how to do something or for step-by-step instructions
- comparative: asking to compare or contrast things
- general: general or conversational questions

Question: {query}

Return only the category name:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip().lower()
            valid_intents = [
                "factual",
                "analytical",
                "procedural",
                "comparative",
                "general",
            ]

            if response_text in valid_intents:
                return response_text
            else:
                # Fallback classification based on keywords
                query_lower = query.lower()
                if any(
                    word in query_lower
                    for word in ["what is", "define", "definition", "who is"]
                ):
                    return "factual"
                elif any(
                    word in query_lower
                    for word in ["how to", "steps", "procedure", "process"]
                ):
                    return "procedural"
                elif any(
                    word in query_lower
                    for word in ["compare", "contrast", "difference", "versus"]
                ):
                    return "comparative"
                elif any(
                    word in query_lower
                    for word in ["why", "analyze", "explain", "reasoning"]
                ):
                    return "analytical"
                else:
                    return "general"

        except Exception as e:
            log_warning(f"Query classification failed: {e}")
            return "general"
