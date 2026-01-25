import re
from dataclasses import dataclass


@dataclass
class QueryClassification:
    """Classification result for a user query."""
    query_type: str
    confidence: float = 0.0


class QueryClassifier:
    """
    Classifies user queries into different types for appropriate handling.
    
    Query Types:
        - Factual: Questions seeking specific facts or definitions
        - Analytical: Questions requiring deeper analysis or explanation
        - Procedural: Questions about how to do something
        - Comparative: Questions comparing two or more things
        - Exploratory: Questions seeking general overview or discussion
    """
    
    QUERY_TYPE_KEYWORDS = {
        "Factual": [
            r"\b(what|who|where|when|which|define|meaning)\b",
            r"\b(is|are|was|were)\b.*\b(definition|called)\b",
        ],
        "Analytical": [
            r"\b(why|how\s+did|significance|importance|role|impact)\b",
            r"\b(analyze|explain|symbolism|represent|meaning|influence)\b",
            r"\b(cultural\s+significance|historical\s+context)\b",
        ],
        "Procedural": [
            r"\b(how\s+to|steps|process|procedure|instructions)\b",
            r"\b(make|cook|create|prepare|perform)\b",
            r"\b(guide|tutorial|method)\b",
        ],
        "Comparative": [
            r"\b(difference|similar|compare|contrast|versus|vs)\b",
            r"\b(alike|unlike|comparison|distinguish)\b",
            r"\b(better|worse|more|less)\s+than\b",
        ],
        "Exploratory": [
            r"\b(overview|describe|tell\s+me\s+about|background)\b",
            r"\b(discuss|elaborate|explore|context)\b",
            r"\b(general|broad|comprehensive)\b",
        ],
    }

    def __init__(self):
        """Initialize the query classifier with compiled regex patterns."""
        self.type_patterns = {
            qtype: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] 
            for qtype, patterns in self.QUERY_TYPE_KEYWORDS.items()
        }

    def classify_query_type(self, query: str) -> tuple[str, float]:
        """
        Classify the type of query based on keywords and patterns.
        
        Args:
            query: The user's query string
            
        Returns:
            Tuple of (query_type, confidence_score)
        """
        scores = {qtype: 0 for qtype in self.type_patterns.keys()}

        # Count pattern matches for each query type
        for qtype, patterns in self.type_patterns.items():
            for pattern in patterns:
                matches = len(pattern.findall(query))
                scores[qtype] += matches

        query_lower = query.lower()

        # Apply additional scoring for specific trigger words
        analytical_triggers = ["how did", "significance", "importance", "role", "impact", "influence"]
        if any(t in query_lower for t in analytical_triggers):
            scores["Analytical"] += 2

        procedural_triggers = ["process", "procedure"]
        if any(t in query_lower for t in procedural_triggers):
            scores["Procedural"] += 2

        comparative_triggers = ["difference", "compare", "contrast", "vs", "versus"]
        if any(t in query_lower for t in comparative_triggers):
            scores["Comparative"] += 2

        # Default to Factual if no matches
        if max(scores.values()) == 0:
            return "Factual", 0.5

        # Select best type with priority ordering
        priority_order = ["Analytical", "Procedural", "Comparative", "Exploratory", "Factual"]
        best_type = sorted(
            scores.items(), 
            key=lambda kv: (-kv[1], priority_order.index(kv[0]))
        )[0][0]

        # Calculate confidence
        total_matches = sum(scores.values())
        confidence = scores[best_type] / total_matches if total_matches > 0 else 0.5

        return best_type, min(confidence, 1.0)

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a user query.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryClassification with type and confidence
        """
        query_type, confidence = self.classify_query_type(query)
        return QueryClassification(query_type=query_type, confidence=confidence)