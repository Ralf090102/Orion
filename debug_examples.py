from app.conversation_memory import FollowUpDetector, ConversationContext

detector = FollowUpDetector()
context = ConversationContext(
    recent_topics=["machine", "learning", "algorithms"],
    recent_sources=[],
    last_user_query="tell me more about the algorithms",
    conversation_summary=None,
    active_references=set(),
)

queries = [
    "what are some examples?",
    "can you give me some examples?",
    "for example?",
    "what are some algorithms?",
]

for query in queries:
    result = detector.detect_query_type(query, context)
    print(f"'{query}' -> {result}")
