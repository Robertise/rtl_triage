#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Common LLM Interface
Defines the unified LLMClient interface with two implementations:
  1. BedrockClient: AWS Bedrock Converse API client for Claude Haiku 4.5
     (global.anthropic.claude-haiku-4-5-20251001-v1:0, ap-southeast-1, max_tokens=8192,
      no temperature/top_p/top_k, tool_choice=auto when tools are provided).
  2. FakeLLM: Deterministic scripted offline LLM client for testing without network/API calls.
"""

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "ap-southeast-1"
DEFAULT_MAX_TOKENS = 8192


class LLMInfraError(RuntimeError):
    """Raised on transient network, connection, timeout, throttling (429), or 5xx Bedrock errors."""


class LLMConfigError(RuntimeError):
    """Raised on configuration or client errors (AccessDenied, ValidationException, invalid model, missing credentials, etc.)."""


def classify_bedrock_exception(exc: Exception) -> Exception:
    """
    Classifies a Bedrock invocation exception into either LLMInfraError or LLMConfigError:
    - LLMInfraError: Network connectivity, timeout, HTTP 429 / Throttling, and HTTP 5xx server errors.
    - LLMConfigError: AccessDenied, ValidationException, invalid model/profile, missing credentials,
      and any unrecognized exceptions.
    Determined strictly via exception type, botocore ClientError code, and HTTP status code.
    Never searches for crude substrings or digit patterns in exception messages.
    """
    exc_type_name = type(exc).__name__

    # 1. Connection / Network / Timeout exceptions (botocore or standard library)
    infra_type_names = {
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
    }
    if exc_type_name in infra_type_names or isinstance(exc, (ConnectionError, TimeoutError)):
        return LLMInfraError(f"Bedrock connection/timeout error ({exc_type_name}): {exc}")

    # 2. Known credential / parameter / service configuration exceptions
    config_type_names = {
        "NoCredentialsError",
        "PartialCredentialsError",
        "ParamValidationError",
        "UnknownServiceError",
    }
    if exc_type_name in config_type_names:
        return LLMConfigError(f"Bedrock configuration error ({exc_type_name}): {exc}")

    # 3. ClientError with structured response dictionary
    if hasattr(exc, "response") and isinstance(exc.response, dict):
        error_info = exc.response.get("Error", {})
        code = str(error_info.get("Code", "")).strip()
        message = str(error_info.get("Message", "")).strip()
        response_meta = exc.response.get("ResponseMetadata", {})
        status_code = int(response_meta.get("HTTPStatusCode", 0))

        throttling_codes = {
            "ThrottlingException",
            "TooManyRequestsException",
            "RequestLimitExceeded",
        }
        server_error_codes = {
            "InternalServerException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
            "ModelErrorException",
        }

        # Check for HTTP 429 or throttling codes
        if status_code == 429 or code in throttling_codes:
            return LLMInfraError(f"Bedrock throttling error (Code={code}, HTTP={status_code}): {message}")

        # Check for HTTP 5xx or server error codes
        if status_code >= 500 or code in server_error_codes:
            return LLMInfraError(f"Bedrock server error (Code={code}, HTTP={status_code}): {message}")

        # All other ClientErrors (AccessDeniedException, ValidationException, ResourceNotFoundException, etc.)
        return LLMConfigError(f"Bedrock client error (Code={code}, HTTP={status_code}): {message}")

    # 4. Unrecognized exceptions default to LLMConfigError
    return LLMConfigError(f"Bedrock configuration/invocation error ({exc_type_name}): {exc}")


@dataclass
class ToolCall:
    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    content_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base interface shared by BedrockClient and FakeLLM."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens

    def build_converse_kwargs(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Constructs the exact Bedrock Converse API payload adhering to preregistration:
        - maxTokens = 8192
        - Strictly NO temperature, topP, or topK
        - toolChoice = {"auto": {}} when tools are provided
        """
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
            },
        }
        if tools:
            kwargs["toolConfig"] = {
                "tools": tools,
                "toolChoice": {"auto": {}},
            }
        return kwargs

    @abstractmethod
    def converse(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Executes a single turn with the model and returns a standardized LLMResponse."""


class BedrockClient(LLMClient):
    """
    Production AWS Bedrock Converse API client.
    Never sends temperature, top_p, or top_k.
    Maps transient infrastructure/network errors to LLMInfraError.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        boto_client: Any = None,
    ) -> None:
        super().__init__(model_id=model_id, region=region, max_tokens=max_tokens)
        self._boto_client = boto_client

    def _get_client(self) -> Any:
        if self._boto_client is None:
            import boto3
            self._boto_client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._boto_client

    def converse(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs = self.build_converse_kwargs(system_prompt=system_prompt, messages=messages, tools=tools)
        client = self._get_client()

        try:
            resp = client.converse(**kwargs)
        except Exception as exc:
            classified = classify_bedrock_exception(exc)
            raise classified from exc

        output_msg = resp.get("output", {}).get("message", {"role": "assistant", "content": []})
        content_blocks = output_msg.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        safe_content_blocks: list[dict[str, Any]] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(str(block["text"]))
                safe_content_blocks.append(block)
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        tool_use_id=str(tu.get("toolUseId", "")),
                        name=str(tu.get("name", "")),
                        input=dict(tu.get("input", {})),
                    )
                )
                safe_content_blocks.append(block)
            # Safe response filtering: Content blocks not text or toolUse (e.g. reasoningContent,
            # guardrails, metadata) are ignored and omitted from safe_content_blocks to prevent
            # breaking the agent conversation loop or causing ValidationException on subsequent turns.

        safe_output_msg = {
            "role": output_msg.get("role", "assistant"),
            "content": safe_content_blocks,
        }

        usage = resp.get("usage", {})
        input_tokens = int(usage.get("inputTokens", 0))
        output_tokens = int(usage.get("outputTokens", 0))
        stop_reason = str(resp.get("stopReason", "end_turn"))

        return LLMResponse(
            content_text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_assistant_message=safe_output_msg,
        )


class FakeLLM(LLMClient):
    """
    Deterministic scripted LLM client for offline testing.
    Accepts a sequence of step specifications (dicts, callables, or Exceptions).
    Records every call payload in `self.calls` for inspection.
    """

    def __init__(
        self,
        script: list[dict[str, Any] | Callable[[str, list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]] | Exception],
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(model_id=model_id, region=region, max_tokens=max_tokens)
        self.script = list(script)
        self.call_index: int = 0
        self.calls: list[dict[str, Any]] = []

    def converse(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs = self.build_converse_kwargs(system_prompt=system_prompt, messages=messages, tools=tools)
        self.calls.append(copy.deepcopy(kwargs))

        if self.call_index >= len(self.script):
            raise RuntimeError(
                f"FakeLLM script exhausted: call #{self.call_index + 1} requested, "
                f"but only {len(self.script)} step(s) were scripted."
            )

        step = self.script[self.call_index]
        self.call_index += 1

        if isinstance(step, Exception):
            if not isinstance(step, (LLMInfraError, LLMConfigError)):
                raise classify_bedrock_exception(step)
            raise step

        if callable(step):
            step_data = step(system_prompt, messages, tools)
        else:
            step_data = step

        if isinstance(step_data, Exception):
            raise step_data

        text = str(step_data.get("text", ""))
        raw_tool_calls = step_data.get("tool_calls", [])
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict[str, Any]] = []

        # Support direct raw_content_blocks simulation to test block filtering
        if "raw_content_blocks" in step_data:
            text_parts: list[str] = []
            for block in step_data["raw_content_blocks"]:
                if "text" in block:
                    text_parts.append(str(block["text"]))
                    content_blocks.append(block)
                elif "toolUse" in block:
                    content_blocks.append(block)
                    tu = block["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            tool_use_id=str(tu.get("toolUseId", "")),
                            name=str(tu.get("name", "")),
                            input=dict(tu.get("input", {})),
                        )
                    )
                # Unexpected blocks (e.g. reasoningContent) are safely filtered out!
            if not text and text_parts:
                text = "\n".join(text_parts)
        else:
            if text:
                content_blocks.append({"text": text})

            for idx, tc in enumerate(raw_tool_calls, start=1):
                t_id = str(tc.get("tool_use_id", f"tooluse_{self.call_index}_{idx}"))
                t_name = str(tc["name"])
                t_input = dict(tc.get("input", {}))
                tool_calls.append(ToolCall(tool_use_id=t_id, name=t_name, input=t_input))
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": t_id,
                            "name": t_name,
                            "input": t_input,
                        }
                    }
                )

        default_stop = "tool_use" if tool_calls else "end_turn"
        stop_reason = str(step_data.get("stop_reason", default_stop))
        input_tokens = int(step_data.get("input_tokens", 1000))
        output_tokens = int(step_data.get("output_tokens", 200))

        raw_msg = {
            "role": "assistant",
            "content": content_blocks,
        }

        return LLMResponse(
            content_text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_assistant_message=raw_msg,
        )
