from src.config import ASSUMPTIONS


# every assumption must carry its full review metadata, no blanks
def test_assumptions_have_review_metadata():
    required = {"value", "source", "confidence", "last_reviewed", "rationale"}

    assert ASSUMPTIONS
    for detail in ASSUMPTIONS.values():
        assert required <= set(detail)
        assert all(detail[k] for k in required)
