"""Pure unit tests for QueryClassifier -- no FastAPI, no fixtures, no I/O.

Contrast with test/api/test_health.py: that layer needs a TestClient and
faked dependencies because it's testing HTTP wiring. This layer is testing
business logic directly, so a bare instance is enough.
"""

import pytest

from src.generation.query_classifier import QueryClassification, QueryClassifier


@pytest.fixture
def classifier():
    return QueryClassifier()


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("What is the capital of the Philippines?", "Factual"),
        ("Why is the Sinulog festival significant?", "Analytical"),
        ("How to cook adobo step by step?", "Procedural"),
        ("What is the difference between Ilocano and Bicolano cuisine?", "Comparative"),
        ("Tell me about Filipino folklore in general", "Exploratory"),
    ],
)
def test_classify_query_type_matches_expected_category(classifier, query, expected_type):
    result = classifier.classify(query)

    assert isinstance(result, QueryClassification)
    assert result.query_type == expected_type


@pytest.mark.unit
def test_classify_defaults_to_factual_with_low_confidence_when_no_keywords_match(classifier):
    result = classifier.classify("asdf qwerty zxcv")

    assert result.query_type == "Factual"
    assert result.confidence == 0.5


@pytest.mark.unit
def test_confidence_is_always_between_zero_and_one(classifier):
    queries = [
        "",
        "how to compare and contrast the significance of why this process matters",
        "What is X",
    ]
    for query in queries:
        result = classifier.classify(query)
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.unit
def test_analytical_beats_factual_on_priority_when_scores_tie(classifier):
    # "what" (Factual) and "impact" (Analytical, +2 trigger boost) both present;
    # Analytical's trigger boost should win despite Factual matching too.
    result = classifier.classify("what is the impact of this event")

    assert result.query_type == "Analytical"
