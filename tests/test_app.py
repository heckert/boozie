import pandas as pd
import pytest

from boozie.app import RoundMatcher


# Fixture to create an instance of the RoundMatcher class for testing
@pytest.fixture
def round_matcher_instance():
    return RoundMatcher()

def test_get_matches(round_matcher_instance):
    # Test case 1: Check if it returns the correct number of matches and labels
    integers = pd.Series([1, 2, 3, 4, 5])
    floats = pd.Series([1.0, 2.0, 2.9, 4.1, 5.5])
    
    result = round_matcher_instance.get_matches(integers, floats)
    
    assert round_matcher_instance.n_matches == 4 
    assert result.tolist() == ["✅", "✅", "✅", "✅", "❌"]

    # Test case 2: Check if it handles empty input correctly
    empty_series = pd.Series([])
    result = round_matcher_instance.get_matches(empty_series, empty_series)
    
    assert round_matcher_instance.n_matches == 0  # No matches
    assert result.empty

    # Test case 3: Check if it handles different data types
    integers = pd.Series([1, 2, 3])
    floats = pd.Series([1.1, 2.2, 3.3])
    
    result = round_matcher_instance.get_matches(integers, floats)
    
    assert round_matcher_instance.n_matches == 3 
    assert result.tolist() == ["✅", "✅", "✅"]