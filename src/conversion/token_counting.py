"""Token counting utilities using Anthropic official API with local fallback."""

import json
from typing import Any, Dict, List, Optional, Union

import httpx
import tiktoken

try:
    from models import Message, SystemContent, Tool, ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult
except ImportError:
    try:
        from models.messages import Message, SystemContent
        from models.tools import Tool
        from models.content_blocks import ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult
    except ImportError:
        # Fallback implementations - basic classes for testing
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        class SystemContent:
            def __init__(self, type, text):
                self.type = type
                self.text = text

        class Tool:
            def __init__(self, name, description=None, input_schema=None):
                self.name = name
                self.description = description
                self.input_schema = input_schema

        class ContentBlockText:
            def __init__(self, text):
                self.text = text

        class ContentBlockImage:
            pass

        class ContentBlockToolUse:
            def __init__(self, name, input):
                self.name = name
                self.input = input

        class ContentBlockToolResult:
            def __init__(self, content):
                self.content = content

try:
    from utils.logging import debug, warning, LogRecord, LogEvent
except ImportError:
    try:
        from utils.logging.handlers import debug, warning, LogRecord, LogEvent
    except ImportError:
        # Fallback implementations
        debug = warning = lambda *args, **kwargs: None
        LogRecord = dict
        class LogEvent:
            COUNT_TOKENS_API_CALL = type('', (), {'value': 'count_tokens_api_call'})()
            COUNT_TOKENS_FALLBACK = type('', (), {'value': 'count_tokens_fallback'})()
            TOOL_INPUT_SERIALIZATION_FAILURE = type('', (), {'value': 'tool_input_serialization_failure'})()
            TOOL_RESULT_SERIALIZATION_FAILURE = type('', (), {'value': 'tool_result_serialization_failure'})()


# Cache for token encoder
_token_encoder_cache: Dict[str, tiktoken.Encoding] = {}


def _get_token_encoder() -> tiktoken.Encoding:
    """Get cached tiktoken encoder for fallback counting."""
    if "cl100k_base" not in _token_encoder_cache:
        _token_encoder_cache["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
    return _token_encoder_cache["cl100k_base"]


def _count_tokens_local_fallback(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
) -> int:
    """Local token counting fallback using tiktoken."""
    enc = _get_token_encoder()
    total_tokens = 0

    # Count system prompt tokens
    if isinstance(system, str):
        total_tokens += len(enc.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, SystemContent) and block.type == "text":
                total_tokens += len(enc.encode(block.text))

    # Count message tokens
    for msg in messages:
        total_tokens += 4  # Base tokens per message
        if msg.role:
            total_tokens += len(enc.encode(msg.role))

        if isinstance(msg.content, str):
            total_tokens += len(enc.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText):
                    total_tokens += len(enc.encode(block.text))
                elif isinstance(block, ContentBlockImage):
                    total_tokens += 768
                elif isinstance(block, ContentBlockToolUse):
                    total_tokens += len(enc.encode(block.name))
                    try:
                        input_str = json.dumps(block.input)
                        total_tokens += len(enc.encode(input_str))
                    except Exception:
                        pass
                elif isinstance(block, ContentBlockToolResult):
                    try:
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            for item in block.content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    content_str += item.get("text", "")
                                else:
                                    content_str += json.dumps(item)
                        else:
                            content_str = json.dumps(block.content)
                        total_tokens += len(enc.encode(content_str))
                    except Exception:
                        pass

    # Count tool tokens
    if tools:
        total_tokens += 2  # Base tokens for tools
        for tool in tools:
            total_tokens += len(enc.encode(tool.name))
            if tool.description:
                total_tokens += len(enc.encode(tool.description))
            try:
                schema_str = json.dumps(tool.input_schema)
                total_tokens += len(enc.encode(schema_str))
            except Exception:
                pass

    return total_tokens


async def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
    provider_manager: Any = None,
) -> int:
    """
    Count tokens for an Anthropic request.

    Tries official API first, falls back to local estimation if API fails.
    """
    if provider_manager is None:
        # No provider manager, use local fallback
        warning(
            LogRecord(
                event=LogEvent.COUNT_TOKENS_FALLBACK.value,
                message="No provider_manager provided, using local fallback",
                request_id=request_id,
            )
        )
        return _count_tokens_local_fallback(messages, system, model_name, tools, request_id)

    # Try official API first
    try:
        # Select a healthy Anthropic provider
        provider = provider_manager.select_healthy_anthropic_provider()

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages],
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = [tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in tools]

        # Get provider headers
        headers = provider_manager.get_provider_headers(provider)

        # Call Anthropic's count_tokens API
        url = f"{provider.base_url}/v1/messages/count_tokens"

        debug(
            LogRecord(
                event=LogEvent.COUNT_TOKENS_API_CALL.value,
                message=f"Calling Anthropic count_tokens API via provider: {provider.name}",
                data={
                    "provider": provider.name,
                    "model": model_name,
                    "url": url
                },
                request_id=request_id,
            )
        )

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            token_count = result["input_tokens"]

            debug(
                LogRecord(
                    event=LogEvent.COUNT_TOKENS_API_CALL.value,
                    message=f"Received accurate token count from Anthropic API: {token_count}",
                    data={
                        "provider": provider.name,
                        "model": model_name,
                        "token_count": token_count
                    },
                    request_id=request_id,
                )
            )

            return token_count

    except Exception as e:
        # API call failed, use local fallback
        warning(
            LogRecord(
                event=LogEvent.COUNT_TOKENS_FALLBACK.value,
                message=f"Count tokens API failed, using local fallback: {type(e).__name__}: {str(e)[:100]}",
                data={
                    "model": model_name,
                    "error": str(e)[:200]
                },
                request_id=request_id,
            )
        )

        return _count_tokens_local_fallback(messages, system, model_name, tools, request_id)