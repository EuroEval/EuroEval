"""Tests for the `leaderboards.queue_parsing` module."""

from leaderboards.queue_parsing import summarise_evaluation_error


class TestSummariseEvaluationError:
    """Tests for `summarise_evaluation_error`."""

    def test_empty_output(self) -> None:
        """An empty or whitespace-only output yields the placeholder."""
        assert summarise_evaluation_error(output="") == "(no output captured)"
        assert summarise_evaluation_error(output="   \n  \n") == "(no output captured)"

    def test_strips_ansi_codes(self) -> None:
        """ANSI colour codes are removed from the summary."""
        output = "\x1b[93mThe model 'x' could not be loaded.\x1b[0m"
        summary = summarise_evaluation_error(output=output)
        assert "\x1b[" not in summary
        assert "could not be loaded" in summary

    def test_surfaces_model_load_error_over_result_noise(self) -> None:
        """The real load error is surfaced despite a FULL_LOG results dump."""
        noise = "\n".join(
            '{"schema_version": "0.2.1", "evaluation_id": "besls/x/1", '
            '"raw_results": "[...]", "evaluation_results": []}'
            for _ in range(50)
        )
        output = (
            "Loading the model 'jinaai/jina-embeddings-v5-text-nano'...\n"
            "The model 'jinaai/jina-embeddings-v5-text-nano' could not be loaded. "
            'The error was ValueError("Unrecognized configuration class '
            "JinaEmbeddingsV5Config for this kind of AutoModel: "
            'AutoModelForSequenceClassification.").\n' + noise
        )
        summary = summarise_evaluation_error(output=output)
        assert "could not be loaded" in summary
        assert "Unrecognized configuration class" in summary
        assert '"schema_version"' not in summary
        assert '"raw_results"' not in summary

    def test_extracts_traceback(self) -> None:
        """A Python traceback is extracted through its exception line."""
        output = (
            "Benchmarking model on Turku NER-fi...\n"
            "Traceback (most recent call last):\n"
            '  File "/x/euroeval", line 10, in <module>\n'
            "    sys.exit(benchmark())\n"
            "httpx.ReadTimeout: The read operation timed out\n"
        )
        summary = summarise_evaluation_error(output=output)
        assert "Traceback (most recent call last):" in summary
        assert "httpx.ReadTimeout: The read operation timed out" in summary

    def test_suppresses_vllm_boilerplate_when_traceback_present(self) -> None:
        """The vLLM 'did not mention' boilerplate is dropped when a traceback exists."""
        output = (
            "Benchmarking model on CoNLL-en...\n"
            "Traceback (most recent call last):\n"
            '  File "/euroeval/src/euroeval/benchmark_modules/vllm.py", '
            "line 1547, in load_model\n"
            "    model = _create_llm()\n"
            '  File "/vllm/engine/llm.py", line 102, in LLM.__init__\n'
            "    self._verify_kv_cache()\n"
            "ValueError: the configured KV-cache size is too small for "
            "sliding-window attention, increase gpu_memory_utilization\n"
            "The model 'CohereLabs/aya-23-35B' could not be loaded, but vLLM did "
            "not mention exactly what happened. Since you're running in verbose "
            "mode, you might see a descriptive error above already. \n"
            "re-running the benchmark with the environment variable `FULL_LOG` "
            "set to `1` to see the full stack trace.\n"
            "Completed 4 benchmarks, and errored 6 benchmarks\n"
        )
        summary = summarise_evaluation_error(output=output)
        assert "configured KV-cache size is too small" in summary
        assert "did not mention exactly what" not in summary
        assert "errored 6 benchmarks" in summary

    def test_includes_errored_summary_line(self) -> None:
        """The euroeval `errored N benchmarks` summary line is included."""
        output = (
            "The model 'm' could not be loaded. The error was ValueError('x').\n"
            "Completed 2 benchmarks, and errored 2 benchmarks\n"
        )
        summary = summarise_evaluation_error(output=output)
        assert "could not be loaded" in summary
        assert "errored 2 benchmarks" in summary

    def test_falls_back_to_tail_without_markers(self) -> None:
        """Without error markers, the tail of the cleaned output is returned."""
        output = "\n".join(f"some log line {i}" for i in range(40))
        summary = summarise_evaluation_error(output=output)
        assert "some log line 39" in summary
        assert "some log line 0" not in summary

    def test_truncates_giant_single_line(self) -> None:
        """A single enormous line is truncated so it can't swamp the summary."""
        giant = "could not be loaded. The error was " + ("x" * 5000)
        summary = summarise_evaluation_error(output=giant)
        assert "…(truncated)" in summary
        assert len(summary) < 1000

    def test_respects_max_chars(self) -> None:
        """The summary never exceeds the requested maximum length."""
        output = "\n".join(
            f"Traceback (most recent call last): frame {i}" for i in range(500)
        )
        summary = summarise_evaluation_error(output=output, max_chars=200)
        assert len(summary) <= 200
