"""Thin OpenAI-compatible chat-completion client (stdlib only, no new deps).

Phase 3 (LLM planner) transport. Provider-neutral at the *protocol* layer (see
``llm_protocol``); this module is the wire: POST ``/chat/completions`` to an
OpenAI-compatible endpoint — the same code path drives a cloud LLM or a local
vLLM 7B (OpenAI-compatible endpoint).

Deliberately stdlib ``urllib`` (~30 lines): ``requests`` is **not** a declared
dependency of this repo and must not enter the execution path. A future
``pydantic_ai`` upgrade replaces this transport + ``planner``, not the protocol.

Default-unreachable: only constructed by ``build_planner`` when
``planner == "llm"``; the flash path never imports this module.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMClientError(Exception):
    """Base for all LLM transport errors (all caught → flash fallback)."""


class LLMConnectionError(LLMClientError):
    """Network-level failure (DNS / timeout / connection refused)."""


class LLMHTTPError(LLMClientError):
    """Non-2xx HTTP response."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"LLM HTTP {status}: {body[:200]}")


class LLMResponseParseError(LLMClientError):
    """2xx response that is not the expected JSON shape."""


class OpenAIClient:
    """Minimal OpenAI-compatible ``/chat/completions`` client.

    ``chat_completion`` returns the raw decoded ``dict``; the planner maps it
    to a ``PlannerDecision``. Every transport/parse failure raises an
    ``LLMClientError`` subclass so the loop degrades to flash.
    """

    def __init__(self, llm_config) -> None:
        self._model = llm_config.model
        self._temperature = llm_config.temperature
        self._timeout = llm_config.timeout
        # key: explicit config wins; else read the named env var.
        self._api_key = llm_config.api_key or os.environ.get(llm_config.api_key_env, "")
        # base_url: OPENAI_BASE_URL env override wins (matches vLLM deployments).
        self._base_url = (os.environ.get("OPENAI_BASE_URL") or llm_config.base_url).rstrip("/")

    def chat_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """POST one chat-completion request and return the decoded JSON object."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise LLMHTTPError(e.code, e.read().decode("utf-8", "replace")) from e
        except urllib.error.URLError as e:
            raise LLMConnectionError(str(e.reason)) from e
        except (TimeoutError, OSError) as e:
            raise LLMConnectionError(str(e)) from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"non-JSON response: {raw[:200]}") from e
        if not isinstance(data, dict):
            raise LLMResponseParseError(f"expected object, got {type(data).__name__}")
        return data
