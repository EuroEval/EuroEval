"""Tests for the `caching_utils` module."""

import pytest

from euroeval.caching_utils import cache_arguments


def test_cache_arguments_basic() -> None:
    """Test basic caching functionality."""
    call_count = 0

    @cache_arguments()
    def expensive_function(x: int, y: int) -> int:
        """A function that we track calls to."""
        nonlocal call_count
        call_count += 1
        return x + y

    # First call should execute the function
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count == 1

    # Second call with same arguments should use cache
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # Should not increment

    # Call with different arguments should execute again
    result3 = expensive_function(2, 3)
    assert result3 == 5
    assert call_count == 2


def test_cache_arguments_with_kwargs() -> None:
    """Test caching with keyword arguments."""
    call_count = 0

    @cache_arguments()
    def function_with_kwargs(x: int, y: int = 10) -> int:
        """A function with keyword arguments."""
        nonlocal call_count
        call_count += 1
        return x + y

    # Test with positional arguments
    result1 = function_with_kwargs(5, 10)
    assert result1 == 15
    assert call_count == 1

    # Test with keyword arguments - same values should use cache
    result2 = function_with_kwargs(x=5, y=10)
    assert result2 == 15
    assert call_count == 1


def test_cache_arguments_specific_args() -> None:
    """Test caching only specific arguments."""
    call_count = 0

    @cache_arguments("x")
    def selective_cache(x: int, y: int) -> int:
        """Function that only caches based on x."""
        nonlocal call_count
        call_count += 1
        return x + y

    # First call
    result1 = selective_cache(1, 2)
    assert result1 == 3
    assert call_count == 1

    # Same x, different y - should still use cache
    result2 = selective_cache(1, 3)
    assert result2 == 3  # Returns cached value (1+2)
    assert call_count == 1

    # Different x - should execute again
    result3 = selective_cache(2, 2)
    assert result3 == 4
    assert call_count == 2


def test_cache_arguments_disable_condition() -> None:
    """Test caching with disable condition."""
    call_count = 0
    should_disable = False

    @cache_arguments(disable_condition=lambda: should_disable)
    def conditional_cache(x: int) -> int:
        """Function with conditional caching."""
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call with cache enabled
    result1 = conditional_cache(5)
    assert result1 == 10
    assert call_count == 1

    # Second call should use cache
    result2 = conditional_cache(5)
    assert result2 == 10
    assert call_count == 1

    # Disable cache
    should_disable = True

    # Third call should not use cache
    result3 = conditional_cache(5)
    assert result3 == 10
    assert call_count == 2


def test_cache_arguments_missing_arg_error() -> None:
    """Test that caching with invalid argument name raises error."""

    @cache_arguments("nonexistent_arg")
    def function_with_cache(x: int) -> int:
        """Function with cache on non-existent arg."""
        return x * 2

    # Should raise ValueError when trying to cache non-existent argument
    with pytest.raises(ValueError, match="Argument nonexistent_arg not found"):
        function_with_cache(5)


def test_cache_arguments_multiple_specific_args() -> None:
    """Test caching multiple specific arguments."""
    call_count = 0

    @cache_arguments("x", "z")
    def multi_arg_cache(x: int, y: int, z: int) -> int:
        """Function that caches based on x and z."""
        nonlocal call_count
        call_count += 1
        return x + y + z

    # First call
    result1 = multi_arg_cache(1, 2, 3)
    assert result1 == 6
    assert call_count == 1

    # Same x and z, different y - should use cache
    result2 = multi_arg_cache(1, 5, 3)
    assert result2 == 6  # Returns cached value
    assert call_count == 1

    # Different z - should execute again
    result3 = multi_arg_cache(1, 2, 4)
    assert result3 == 7
    assert call_count == 2
