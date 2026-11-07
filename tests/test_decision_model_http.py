"""HTTP-level tests for the typed decision-model transport."""

import dataclasses
import json
import threading
import typing as t
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from euroeval.benchmark_modules.decision_model import DecisionModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig


def test_local_system_one_request_returns_classification_scores(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
) -> None:
    """Send typed classification to a local endpoint and use direct probabilities."""
    labels = list(dataset_config.prompt_label_mapping.values())
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        """Return a typed choice response and retain the request for assertions."""

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "model": "org/typed",
                    "answers": {
                        "name": {
                            "probabilities": {
                                labels[0]: 0.8,
                                labels[1]: 0.1,
                                labels[2]: 0.1,
                            }
                        }
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep request content out of test logs."""

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config_file = NamedTemporaryFile(mode="w", delete=False)
        json.dump({"format_id": "onejev-slot-v2", "package": "flide"}, config_file)
        config_file.close()
        monkeypatch.setattr(
            "euroeval.benchmark_modules.decision_model.hf_hub_download",
            lambda **_kwargs: config_file.name,
        )
        model_config = DecisionModel.get_model_config(
            model_id="org/typed", benchmark_config=benchmark_config
        )
        model = DecisionModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=dataclasses.replace(
                benchmark_config, api_base=f"http://127.0.0.1:{server.server_port}"
            ),
            log_metadata=False,
        )

        result = model.generate(inputs={"text": ["A sample"]})
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        Path(config_file.name).unlink(missing_ok=True)

    assert result.sequences == [labels[0]]
    assert result.scores is not None
    request = requests[0]
    questions = t.cast(dict[str, object], request["questions"])
    question = t.cast(dict[str, object], questions["name"])
    assert request["model"] == "org/typed"
    assert "state" in request
    assert question["type"] == "choice"
    assert "Authorization" not in requests[0]
