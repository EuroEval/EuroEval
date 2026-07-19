"""Tests for the `leaderboards.model_metadata` module.

These tests verify that model metadata lookups correctly handle validation
and zero-shot suffixes (e.g., ``(val)``, ``(zero-shot)``) by normalising them
to the base model ID before cache operations.

Also tests that cached open metadata (including False) is preserved without
unnecessary Hugging Face API calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from leaderboards.cache import Cache, _is_hf_url_for_model
from leaderboards.constants import (
    TRAINED_FROM_SCRATCH_PATTERNS,
    UNKNOWN_RESULTS_FILENAME,
)
from leaderboards.model_metadata import (
    add_missing_entries,
    is_commercially_licensed,
    is_merge,
    is_open,
    is_trained_from_scratch,
)


def _record(model_name: str, generative: bool = True) -> dict:
    """Create a minimal record for testing.

    Args:
        model_name:
            The model name for the record.
        generative (optional):
            Whether the model is generative. Defaults to True.

    Returns:
        A minimal record dictionary.
    """
    return {
        "model_info": {"name": model_name, "additional_details": {}},
        "generative": generative,
    }


class TestValSuffixMetadataLookup:
    """Tests that validation/zero-shot suffixes do not break metadata lookups."""

    def test_is_commercially_licensed_uses_base_model_cache(self) -> None:
        """A model with (val) suffix should use cached metadata from base model.

        Regression test for issue where ``Qwen/Qwen3.6-27B-FP8 (val)`` was
        treated as an invalid HF repo and metadata defaulted to false instead
        of using the cached value for ``Qwen/Qwen3.6-27B-FP8``.
        """
        cache = Cache()
        # Pre-populate cache with base model metadata
        cache.commercially_licensed["Qwen/Qwen3.6-27B-FP8"] = True

        # Record with (val) suffix should use the base model's cached value
        record = _record(model_name="Qwen/Qwen3.6-27B-FP8 (val)")
        result = is_commercially_licensed(record=record, cache=cache)

        assert result is True
        # The cache key should be the normalised model ID, not the suffixed one
        assert "Qwen/Qwen3.6-27B-FP8" in cache.commercially_licensed

    def test_is_open_uses_base_model_cache(self) -> None:
        """A model with (val) suffix should use cached openness from base model."""
        cache = Cache()
        cache.open["meta-llama/Llama-3.1-8B"] = True

        record = _record(model_name="meta-llama/Llama-3.1-8B (val)")
        result = is_open(record=record, cache=cache)

        assert result is True

    def test_is_merge_uses_base_model_cache(self) -> None:
        """A model with (zero-shot) suffix should use cached merge status."""
        cache = Cache()
        cache.merge["HuggingFaceH4/zephyr-7b-beta"] = False

        record = _record(model_name="HuggingFaceH4/zephyr-7b-beta (zero-shot)")
        result = is_merge(record=record, cache=cache)

        assert result is False

    def test_is_trained_from_scratch_uses_normalised_model_cache(self) -> None:
        """A model with (val) suffix should use cached scratch-trained status."""
        cache = Cache()
        cache.trained_from_scratch["mistralai/Mistral-7B-v0.1"] = True

        record = _record(model_name="mistralai/Mistral-7B-v0.1 (val)")
        result = is_trained_from_scratch(
            record=record, trained_from_scratch_patterns=[], cache=cache
        )

        assert result is True

    def test_is_trained_from_scratch_uses_exact_model_cache(self) -> None:
        """Hyphenated model families must not share scratch-trained status."""
        cache = Cache()
        cache.trained_from_scratch["openeurollm/oellm-1b"] = False
        cache.trained_from_scratch["openeurollm/oellm-9b-128k-theta32m-v3"] = True

        record = _record(model_name="openeurollm/oellm-9b-128k-theta32m-v3")
        result = is_trained_from_scratch(
            record=record, trained_from_scratch_patterns=[], cache=cache
        )

        assert result is True

    @patch("leaderboards.model_metadata.input")
    def test_is_trained_from_scratch_does_not_use_hyphen_prefix_cache(
        self, mock_input: MagicMock
    ) -> None:
        """A sibling model prefix must not suppress the manual prompt."""
        cache = Cache()
        cache.open["openeurollm/oellm-9b-128k-theta32m-v3"] = True
        cache.trained_from_scratch["openeurollm/oellm-1b"] = False
        mock_input.return_value = "y"

        record = _record(model_name="openeurollm/oellm-9b-128k-theta32m-v3")
        result = is_trained_from_scratch(
            record=record, trained_from_scratch_patterns=[], cache=cache
        )

        assert result is True
        mock_input.assert_called_once()
        assert (
            cache.trained_from_scratch["openeurollm/oellm-9b-128k-theta32m-v3"] is True
        )

    @patch("leaderboards.model_metadata.HfApi")
    def test_is_merge_looks_up_base_model_on_hf(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """HfApi.model_info should be called with base model ID, not suffixed one.

        This test verifies that when metadata isn't cached, the HF API is called
        with the normalised model ID (without suffix) rather than the raw ID.
        """
        cache = Cache()
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Model info with merge tags
        mock_model_info = MagicMock()
        mock_model_info.tags = ["merge", "mergekit"]
        mock_api.model_info.return_value = mock_model_info

        # Record with (val) suffix
        record = _record(model_name="openchat/openchat-3.5-1210 (val)")

        result = is_merge(record=record, cache=cache)

        # Should return True (has merge tag)
        assert result is True
        # HfApi.model_info should have been called with base model ID
        mock_api.model_info.assert_called_once_with(
            repo_id="openchat/openchat-3.5-1210"
        )
        # Cache should store under normalised key
        assert "openchat/openchat-3.5-1210" in cache.merge

    @patch("leaderboards.model_metadata.HfApi")
    def test_is_open_looks_up_base_model_on_hf(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """HfApi.model_info should be called with base model ID for openness check."""
        cache = Cache()
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Model exists on HF (no exception raised)
        mock_api.model_info.return_value = MagicMock()

        record = _record(model_name="google/gemma-2-9b (zero-shot, val)")

        result = is_open(record=record, cache=cache)

        assert result is True
        mock_api.model_info.assert_called_once_with(repo_id="google/gemma-2-9b")

    @patch("leaderboards.model_metadata.HfApi")
    def test_is_open_returns_false_for_nonexistent_base_model(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Non-existent base model should return False for openness.

        When a model with suffix isn't found on HF (using base ID), it should
        be treated as closed, not raise an error.
        """
        cache = Cache()
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # RepositoryNotFoundError requires a response object
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not found"
        mock_response.request = Mock()
        mock_response.request.url = "https://huggingface.co/api/models/fake"

        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Model not found", response=mock_response
        )

        record = _record(model_name="unknown-org/unknown-model (val)")

        result = is_open(record=record, cache=cache)

        assert result is False
        mock_api.model_info.assert_called_once_with(repo_id="unknown-org/unknown-model")

    def test_combined_zero_shot_val_suffix(self) -> None:
        """Combined (zero-shot, val) suffix should also normalise correctly."""
        cache = Cache()
        cache.commercially_licensed["Qwen/Qwen2.5-72B-Instruct"] = True

        record = _record(model_name="Qwen/Qwen2.5-72B-Instruct (zero-shot, val)")
        result = is_commercially_licensed(record=record, cache=cache)

        assert result is True

    @patch("leaderboards.model_metadata.HfApi")
    def test_merge_detection_ignores_suffix(self, mock_hf_api_class: MagicMock) -> None:
        """Merge detection via HF tags should work regardless of suffix."""
        cache = Cache()
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Model without merge tags
        mock_model_info = MagicMock()
        mock_model_info.tags = ["pytorch", "safetensors"]
        mock_api.model_info.return_value = mock_model_info

        record = _record(model_name="microsoft/Phi-3-mini-4k-instruct (val)")

        result = is_merge(record=record, cache=cache)

        assert result is False
        mock_api.model_info.assert_called_once_with(
            repo_id="microsoft/Phi-3-mini-4k-instruct"
        )


class TestTrainedFromScratchPatterns:
    """Tests for trained_from_scratch pattern matching with suffixes."""

    def test_pattern_matching_strips_suffix_first(self) -> None:
        """Pattern matching should happen on normalised model ID.

        This ensures that patterns like ``^fresh/.*`` match even when the
        record has a (val) suffix.
        """
        cache = Cache()
        # Pattern for "fresh" models
        patterns = [re.compile(r"^fresh/.*")]

        record = _record(model_name="fresh/my-custom-model (val)")
        result = is_trained_from_scratch(
            record=record, trained_from_scratch_patterns=patterns, cache=cache
        )

        # Should match the pattern and return True without prompting
        assert result is True

    def test_pre_existing_curated_patterns_still_match(self) -> None:
        """Regression test: pre-existing curated patterns must be preserved.

        This ensures that curated metadata patterns (e.g. Qwen, Google, Meta)
        were not accidentally removed. The reviewer objected only to newly
        invented inference rules, not to deleting existing curated patterns.
        """
        cache = Cache()

        # Test a pre-existing curated family (Qwen)
        record_qwen = _record(model_name="Qwen/Qwen3.6-27B-FP8")
        result_qwen = is_trained_from_scratch(
            record=record_qwen,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )
        assert result_qwen is True, "Qwen models should match curated patterns"

        # Test GLM explicit patterns
        record_glm_zai = _record(model_name="zai-org/GLM-Edge-1.5B")
        result_glm_zai = is_trained_from_scratch(
            record=record_glm_zai,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )
        assert result_glm_zai is True, "zai-org/GLM should match explicit pattern"

        record_glm_gadfly = _record(model_name="GadflyII/GLM-4-9B-Base")
        result_glm_gadfly = is_trained_from_scratch(
            record=record_glm_gadfly,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )
        assert result_glm_gadfly is True, "GadflyII/GLM should match explicit pattern"

        # Test other pre-existing curated families
        record_google = _record(model_name="google/gemma-2b")
        result_google = is_trained_from_scratch(
            record=record_google,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )
        assert result_google is True, "google models should match curated patterns"

        record_meta = _record(model_name="meta-llama/Llama-3.2-1B")
        result_meta = is_trained_from_scratch(
            record=record_meta,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )
        assert result_meta is True, "meta-llama models should match curated patterns"


class TestCommercialLicenceInference:
    """Tests for best-effort commercial licence inference from HF model info."""

    @patch("leaderboards.model_metadata.HfApi")
    def test_permissive_licence_inferred_without_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Model with permissive licence (e.g. MIT) returns True without input().

        Regression test: BAAI/bge-m3 with licence:mit should not prompt.
        """
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Mock model_info to return MIT licence
        mock_model_info = MagicMock()
        mock_model_info.tags = ["transformers", "license:mit", "safetensors"]
        mock_api.model_info.return_value = mock_model_info

        # Record for BAAI/bge-m3-like model
        record = _record(model_name="BAAI/bge-m3")
        cache = Cache()

        # Should return True without calling input()
        result = is_commercially_licensed(record=record, cache=cache)

        assert result is True
        assert cache.commercially_licensed["BAAI/bge-m3"] is True
        mock_api.model_info.assert_called_once_with(repo_id="BAAI/bge-m3")

    @patch("leaderboards.model_metadata.HfApi")
    def test_permissive_licence_variants(self, mock_hf_api_class: MagicMock) -> None:
        """Various permissive licences (apache-2.0, BSD, etc.) infer True."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_model_info = MagicMock()
        mock_model_info.tags = ["license:apache-2.0"]
        mock_api.model_info.return_value = mock_model_info

        record = _record(model_name="org/apache-model")
        cache = Cache()

        result = is_commercially_licensed(record=record, cache=cache)

        assert result is True

    @patch("leaderboards.model_metadata.HfApi")
    def test_non_permissive_licence_falls_through_to_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Non-permissive licence (e.g. llama-3) should fall through to prompt."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_model_info = MagicMock()
        mock_model_info.tags = ["license:llama-3"]
        mock_api.model_info.return_value = mock_model_info

        record = _record(model_name="meta-llama/Llama-3")
        cache = Cache()

        # Patch input to return "n" (not commercially licensed)
        with patch("leaderboards.model_metadata.input", return_value="n"):
            result = is_commercially_licensed(record=record, cache=cache)

        assert result is False
        assert cache.commercially_licensed["meta-llama/Llama-3"] is False

    @patch("leaderboards.model_metadata.HfApi")
    def test_no_licence_tag_falls_through_to_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Model without licence tag should fall through to prompt."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_model_info = MagicMock()
        mock_model_info.tags = ["transformers", "safetensors"]  # No licence tag
        mock_api.model_info.return_value = mock_model_info

        record = _record(model_name="org/model-no-licence")
        cache = Cache()

        with patch("leaderboards.model_metadata.input", return_value="y"):
            result = is_commercially_licensed(record=record, cache=cache)

        assert result is True

    @patch("leaderboards.model_metadata.HfApi")
    def test_hf_repository_not_found_falls_through_to_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """RepositoryNotFoundError should fall through to prompt, not crash."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # RepositoryNotFoundError requires a response object
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not found"
        mock_response.request = Mock()
        mock_response.request.url = "https://huggingface.co/api/models/fake"

        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Model not found", response=mock_response
        )

        record = _record(model_name="unknown-org/unknown-model")
        cache = Cache()

        with patch("leaderboards.model_metadata.input", return_value="y"):
            result = is_commercially_licensed(record=record, cache=cache)

        assert result is True
        mock_api.model_info.assert_called_once_with(repo_id="unknown-org/unknown-model")

    @patch("leaderboards.model_metadata.HfApi")
    def test_httpx_error_falls_through_to_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Httpx errors should fall through to prompt, not crash."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_api.model_info.side_effect = httpx.ConnectError("Connection failed")

        record = _record(model_name="org/httpx-error-model")
        cache = Cache()

        with patch("leaderboards.model_metadata.input", return_value="n"):
            result = is_commercially_licensed(record=record, cache=cache)

        assert result is False

    @patch("leaderboards.model_metadata.HfApi")
    def test_gated_repo_error_falls_through_to_prompt(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """GatedRepoError should fall through to prompt, not crash."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # GatedRepoError requires a response object
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_response.text = "Gated"
        mock_response.request = Mock()
        mock_response.request.url = "https://huggingface.co/api/models/fake"

        mock_api.model_info.side_effect = GatedRepoError(
            "Gated repo", response=mock_response
        )

        record = _record(model_name="meta-llama/Llama-2-7b")
        cache = Cache()

        with patch("leaderboards.model_metadata.input", return_value="y"):
            result = is_commercially_licensed(record=record, cache=cache)

        assert result is True

    @patch("leaderboards.model_metadata.HfApi")
    def test_licence_lookup_is_cached(self, mock_hf_api_class: MagicMock) -> None:
        """HF licence lookup should be cached to avoid repeated API calls."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_model_info = MagicMock()
        mock_model_info.tags = ["license:mit"]
        mock_api.model_info.return_value = mock_model_info

        record = _record(model_name="BAAI/bge-m3")
        cache = Cache()

        # First call - should call HF API
        result1 = is_commercially_licensed(record=record, cache=cache)
        assert result1 is True
        assert mock_api.model_info.call_count == 1

        # Second call with same model - should use cache
        result2 = is_commercially_licensed(record=record, cache=cache)
        assert result2 is True
        assert mock_api.model_info.call_count == 1  # Not called again

    @patch("leaderboards.model_metadata.HfApi")
    def test_error_result_is_cached_to_avoid_repeated_failures(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """HF lookup errors should be cached to avoid repeated API calls.

        After an error, the user is prompted. Their answer is cached, so
        subsequent calls don't re-prompt or re-try the HF API.
        """
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_api.model_info.side_effect = httpx.ConnectError("Connection failed")

        record = _record(model_name="org/error-model")
        cache = Cache()

        # First call - should try HF API and fail, then prompt
        with patch("leaderboards.model_metadata.input", return_value="y") as mock_input:
            result1 = is_commercially_licensed(record=record, cache=cache)
        assert result1 is True
        assert mock_api.model_info.call_count == 1
        assert mock_input.call_count == 1

        # Second call - should use cached user decision, no HF lookup or prompt
        with patch("leaderboards.model_metadata.input", return_value="n") as mock_input:
            result2 = is_commercially_licensed(record=record, cache=cache)
        assert result2 is True  # Cached from first call
        assert mock_api.model_info.call_count == 1  # Not called again
        assert mock_input.call_count == 0  # Not prompted again


class TestCachePriority:
    """Tests for cache priority in is_open().

    These tests ensure that:
    1. Cached open values (True and False) are returned without HF API calls.
    2. Base-model cache broadening does not propagate values across variants.
    3. Non-HF/nonexistent model false remains false.
    4. Cached False is preserved, not automatically repaired.
    """

    def test_cached_false_is_preserved_without_hf_call(self) -> None:
        """Cached open=False is returned without HF API call."""
        # Record with open=False and HF URL
        record = {
            "model_info": {
                "name": "Qwen/Qwen3-32B#no-thinking",
                "additional_details": {
                    "open": False,
                    "model_url": "https://hf.co/Qwen/Qwen3-32B",
                },
            },
            "generative": True,
        }
        cache = Cache()
        # Pre-populate cache with False value
        # Note: split_model_id() strips '#no-thinking', so cache key is 'Qwen/Qwen3-32B'
        cache.open["Qwen/Qwen3-32B"] = False

        with patch("leaderboards.model_metadata.HfApi") as mock_hf_api_class:
            result = is_open(record=record, cache=cache)

            # Cached False is returned, no HF call
            assert result is False
            assert cache.open["Qwen/Qwen3-32B"] is False
            mock_hf_api_class.assert_not_called()

    @patch("leaderboards.model_metadata.HfApi")
    def test_base_model_broadening_does_not_propagate_false(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Cached false for Qwen/Qwen3-0.6B should not affect Qwen/Qwen3-32B.

        This tests that prefix broadening is NOT used for openness, preventing
        propagation of stale values across unrelated model variants.
        HF lookup should succeed for Qwen3-32B and return True.
        """
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        # Model exists on HF (no exception raised)
        mock_api.model_info.return_value = MagicMock()

        cache = Cache()
        # Cache False for a different Qwen3 variant
        cache.open["Qwen/Qwen3-0.6B"] = False

        # Record for Qwen3-32B (no HF URL, not in cache)
        record = {
            "model_info": {"name": "Qwen/Qwen3-32B", "additional_details": {}},
            "generative": True,
        }

        # Should do HF lookup (not use cached False from Qwen/Qwen3-0.6B)
        result = is_open(record=record, cache=cache)

        # Result is True because HF lookup succeeds
        assert result is True
        # Verify model_info was called with the exact repo_id
        mock_api.model_info.assert_called_once_with(repo_id="Qwen/Qwen3-32B")
        # Verify that Qwen/Qwen3-32B now has its own cache entry (not inherited)
        assert "Qwen/Qwen3-32B" in cache.open
        assert cache.open["Qwen/Qwen3-32B"] is True
        # Verify Qwen/Qwen3-0.6B cache is still there unchanged
        assert cache.open["Qwen/Qwen3-0.6B"] is False

    @patch("leaderboards.model_metadata.HfApi")
    def test_non_hf_model_false_remains_false(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """Non-HF/nonexistent model should remain False."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        # Model does NOT exist on HF (RepositoryNotFoundError raised)
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not Found"
        mock_response.request = Mock()
        mock_response.request.url = "https://huggingface.co/api/models/fake"
        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Repository Not Found", response=mock_response
        )

        record = {
            "model_info": {"name": "openai/gpt-4", "additional_details": {}},
            "generative": True,
        }
        cache = Cache()

        result = is_open(record=record, cache=cache)

        assert result is False
        assert cache.open["openai/gpt-4"] is False

    def test_is_hf_url_for_model_detects_hf_urls(self) -> None:
        """Test the _is_hf_url_for_model helper function."""
        assert (
            _is_hf_url_for_model("https://hf.co/Qwen/Qwen3-32B", "Qwen/Qwen3-32B")
            is True
        )
        assert (
            _is_hf_url_for_model(
                "https://huggingface.co/Qwen/Qwen3-32B", "Qwen/Qwen3-32B"
            )
            is True
        )
        assert (
            _is_hf_url_for_model(
                "https://hf.co/meta-llama/Llama-3.1-8B", "Qwen/Qwen3-32B"
            )
            is False
        )
        assert _is_hf_url_for_model("https://openai.com/gpt-4", "openai/gpt-4") is False
        assert (
            _is_hf_url_for_model("https://api.openai.com/gpt-4", "openai/gpt-4")
            is False
        )
        # Boundary test: repo2 should NOT match repo
        assert _is_hf_url_for_model("https://hf.co/org/repo2", "org/repo") is False
        assert (
            _is_hf_url_for_model("https://huggingface.co/org/repo2", "org/repo")
            is False
        )
        # But exact match should work
        assert _is_hf_url_for_model("https://hf.co/org/repo", "org/repo") is True

    @patch("leaderboards.model_metadata.HfApi")
    def test_cached_true_is_trusted(self, mock_hf_api_class: MagicMock) -> None:
        """Cached open=True should be returned without HF lookup."""
        cache = Cache()
        # split_model_id() strips '#no-thinking', so cache key is 'Qwen/Qwen3-32B'
        cache.open["Qwen/Qwen3-32B"] = True

        record = {
            "model_info": {
                "name": "Qwen/Qwen3-32B#no-thinking",
                "additional_details": {},
            },
            "generative": True,
        }

        result = is_open(record=record, cache=cache)

        assert result is True
        # Verify HF API was NOT called (cached True is trusted)
        mock_hf_api_class.assert_not_called()

    def test_cache_from_records_caches_false_with_hf_url(self) -> None:
        """Cache._from_records() should cache open=False even when HF URL exists."""
        # Create a record with open=False and HF URL
        record: dict[str, object] = {
            "model_info": {
                "name": "org/repo",
                "additional_details": {
                    "open": False,
                    "model_url": "https://hf.co/org/repo",
                },
            },
            "generative": True,
        }

        cache = Cache._from_records(records=[record], desc="Test")

        # Should cache False even with HF URL (cache priority)
        assert cache.open["org/repo"] is False

    def test_cache_from_records_caches_false_without_hf_url(self) -> None:
        """Cache._from_records() should cache open=False when no HF URL."""
        # Create a record with open=False and no model_url
        record: dict[str, object] = {
            "model_info": {
                "name": "openai/gpt-4",
                "additional_details": {"open": False},
            },
            "generative": True,
        }

        cache = Cache._from_records(records=[record], desc="Test")

        # Should cache False because there's no HF URL
        assert cache.open["openai/gpt-4"] is False

    def test_cache_from_records_caches_true_with_hf_url(self) -> None:
        """Cache._from_records() should cache open=True even with HF URL."""
        # Create a record with open=True and HF URL
        record: dict[str, object] = {
            "model_info": {
                "name": "org/repo",
                "additional_details": {
                    "open": True,
                    "model_url": "https://hf.co/org/repo",
                },
            },
            "generative": True,
        }

        cache = Cache._from_records(records=[record], desc="Test")

        # Should cache True
        assert cache.open["org/repo"] is True

    def test_add_missing_entries_preserves_existing_open_false(self) -> None:
        """add_missing_entries should preserve existing open=False, not recheck."""
        # Record with existing open=False and HF URL
        # Pre-populate other fields to avoid interactive prompts
        record = {
            "model_info": {
                "name": "Qwen/Qwen3-32B",
                "additional_details": {
                    "open": False,
                    "model_url": "https://hf.co/Qwen/Qwen3-32B",
                    "generative_type": "base",
                    "merge": False,
                    "commercially_licensed": False,
                    "trained_from_scratch": False,
                },
            },
            "generative": True,
            "eval_library": {
                "additional_details": {"validation_split": False, "few_shot": True}
            },
        }
        cache = Cache()

        with patch("leaderboards.model_metadata.HfApi") as mock_hf_api_class:
            result = add_missing_entries(
                record=record,
                trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
                cache=cache,
            )

            # open should remain False, no HF call
            assert result["model_info"]["additional_details"]["open"] is False
            mock_hf_api_class.assert_not_called()

    def test_add_missing_entries_preserves_existing_open_false_with_param_suffix(
        self,
    ) -> None:
        """add_missing_entries preserves existing open=False with param suffix."""
        # Record with existing open=False, HF URL, and parameterised model ID
        # Pre-populate other fields to avoid interactive prompts
        record = {
            "model_info": {
                "name": "Qwen/Qwen3-32B#no-thinking",
                "additional_details": {
                    "open": False,
                    "model_url": "https://hf.co/Qwen/Qwen3-32B",
                    "generative_type": "base",
                    "merge": False,
                    "commercially_licensed": False,
                    "trained_from_scratch": False,
                },
            },
            "generative": True,
            "eval_library": {
                "additional_details": {"validation_split": False, "few_shot": True}
            },
        }
        cache = Cache()

        with patch("leaderboards.model_metadata.HfApi") as mock_hf_api_class:
            result = add_missing_entries(
                record=record,
                trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
                cache=cache,
            )

            # open should remain False, no HF call
            assert result["model_info"]["additional_details"]["open"] is False
            mock_hf_api_class.assert_not_called()

    @patch("leaderboards.model_metadata.HfApi")
    def test_add_missing_entries_preserves_non_hf_false(
        self, mock_hf_api_class: MagicMock
    ) -> None:
        """add_missing_entries should preserve open=False for non-HF models."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        # Model does NOT exist on HF
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not Found"
        mock_response.request = Mock()
        mock_response.request.url = "https://huggingface.co/api/models/fake"
        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Repository Not Found", response=mock_response
        )

        # Record with open=False and non-HF URL
        # Pre-populate to avoid interactive prompts
        record = {
            "model_info": {
                "name": "openai/gpt-4",
                "additional_details": {
                    "open": False,
                    "model_url": "https://openai.com/gpt-4",
                    "generative_type": "base",
                    "merge": False,
                    "commercially_licensed": False,
                    "trained_from_scratch": False,
                },
            },
            "generative": True,
            "eval_library": {
                "additional_details": {"validation_split": False, "few_shot": True}
            },
        }
        cache = Cache()

        result = add_missing_entries(
            record=record,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=cache,
        )

        # open should remain False (non-HF URL)
        assert result["model_info"]["additional_details"]["open"] is False


def test_cache_ignores_unknown_jsonl(tmp_path: Path) -> None:
    """Stale unknown.jsonl records must not override per-model metadata."""
    rich_record = {
        "model_info": {
            "name": "Qwen/Qwen3.6-27B-FP8",
            "additional_details": {
                "open": True,
                "commercially_licensed": True,
                "trained_from_scratch": False,
            },
        }
    }
    stale_record = {
        "model_info": {
            "name": "Qwen/Qwen3.6-27B-FP8",
            "additional_details": {
                "open": False,
                "commercially_licensed": False,
                "trained_from_scratch": False,
            },
        }
    }

    (tmp_path / "Qwen_Qwen3.6-27B-FP8.jsonl").write_text(
        json.dumps(rich_record) + "\n", encoding="utf-8"
    )
    (tmp_path / UNKNOWN_RESULTS_FILENAME).write_text(
        json.dumps(stale_record) + "\n", encoding="utf-8"
    )

    cache = Cache.from_results_dir(results_dir=tmp_path)

    assert cache.open["Qwen/Qwen3.6-27B-FP8"] is True
    assert cache.commercially_licensed["Qwen/Qwen3.6-27B-FP8"] is True
    assert cache.trained_from_scratch["Qwen/Qwen3.6-27B-FP8"] is False
