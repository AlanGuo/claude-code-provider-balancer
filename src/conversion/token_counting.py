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
            # Only count text blocks
            if hasattr(block, 'type') and block.type == "text":
                if hasattr(block, 'text') and isinstance(block.text, str):
                    total_tokens += len(enc.encode(block.text))
                elif hasattr(block, 'text') and isinstance(block.text, list):
                    # Handle text as array
                    for text_part in block.text:
                        total_tokens += len(enc.encode(text_part or ""))

    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(enc.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText) or (hasattr(block, 'type') and block.type == "text"):
                    text = block.text if hasattr(block, 'text') else ""
                    total_tokens += len(enc.encode(text))
                elif isinstance(block, ContentBlockImage) or (hasattr(block, 'type') and block.type == "image"):
                    # Estimate for images
                    total_tokens += 768
                elif isinstance(block, ContentBlockToolUse) or (hasattr(block, 'type') and block.type == "tool_use"):
                    # Only count input, not name (matches TypeScript)
                    try:
                        input_data = block.input if hasattr(block, 'input') else {}
                        input_str = json.dumps(input_data)
                        total_tokens += len(enc.encode(input_str))
                    except Exception:
                        pass
                elif isinstance(block, ContentBlockToolResult) or (hasattr(block, 'type') and block.type == "tool_result"):
                    try:
                        content_str = ""
                        content = block.content if hasattr(block, 'content') else ""
                        if isinstance(content, str):
                            content_str = content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    content_str += item.get("text", "")
                                else:
                                    content_str += json.dumps(item)
                        else:
                            content_str = json.dumps(content)
                        total_tokens += len(enc.encode(content_str))
                    except Exception:
                        pass

    # Count tool tokens
    if tools:
        for tool in tools:
            # Combine name and description like TypeScript does
            if hasattr(tool, 'description') and tool.description:
                combined = tool.name + tool.description
                total_tokens += len(enc.encode(combined))
            else:
                total_tokens += len(enc.encode(tool.name))

            # Count input_schema
            if hasattr(tool, 'input_schema') and tool.input_schema:
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
    original_headers: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Count tokens for an Anthropic request.

    Tries official API first (with intelligent availability checking),
    falls back to local estimation if API fails or is marked as unavailable.
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

        # 检查count_tokens API是否可用
        if not provider_manager.is_count_tokens_api_available(provider.name):
            # API标记为不可用，直接使用本地fallback
            debug(
                LogRecord(
                    event=LogEvent.COUNT_TOKENS_USING_CACHED_STATUS.value,
                    message=f"Provider {provider.name} count_tokens API is marked unavailable, using local fallback directly",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "model": model_name,
                        "reason": "api_marked_unavailable"
                    }
                )
            )
            return _count_tokens_local_fallback(messages, system, model_name, tools, request_id)

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
        headers = provider_manager.get_provider_headers(provider, original_headers)

        # Call Anthropic's count_tokens API
        url = f"{provider.base_url}/v1/messages/count_tokens?beta=true"

        # Check if there's a timeout override for count_tokens requests
        timeout_override = provider_manager.get_count_tokens_timeout_override()
        if timeout_override is not None:
            # Use overridden timeout
            timeout_config = httpx.Timeout(
                connect=timeout_override,
                read=timeout_override,
                write=timeout_override,
                pool=timeout_override
            )
        else:
            # Use default non-streaming timeouts
            http_timeouts = provider_manager.get_timeouts_for_request(False)
            timeout_config = httpx.Timeout(
                connect=http_timeouts['connect_timeout'],
                read=http_timeouts['read_timeout'],
                write=http_timeouts['read_timeout'],
                pool=http_timeouts['pool_timeout']
            )

        # Configure proxy if specified
        proxy_config = None
        if provider.proxy:
            proxy_config = provider.proxy

        # Manually serialize JSON and set content-length
        headers = dict(headers) if headers else {}
        json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers['content-type'] = 'application/json'
        headers['content-length'] = str(len(json_data))

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

        async with httpx.AsyncClient(timeout=timeout_config, proxy=proxy_config) as client:
            response = await client.post(url, content=json_data, headers=headers)
            response.raise_for_status()
            result = response.json()
            token_count = result["input_tokens"]

            # 标记API调用成功
            provider_manager.mark_count_tokens_api_success(provider.name, request_id)

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
        # API call failed, mark as failed and use local fallback
        error_data = {
            "model": model_name,
            "error_type": type(e).__name__,
            "error": str(e)[:500]
        }

        # Add more details for HTTP errors
        if hasattr(e, 'response'):
            error_data['status_code'] = getattr(e.response, 'status_code', None)
            error_data['url'] = str(getattr(e.response, 'url', url))
            try:
                error_data['response_body'] = e.response.text[:500]
            except:
                pass

        # 标记count_tokens API失败
        if provider_manager and 'provider' in locals():
            provider_manager.mark_count_tokens_api_failed(provider.name, request_id)

        warning(
            LogRecord(
                event=LogEvent.COUNT_TOKENS_FALLBACK.value,
                message="Count tokens API failed, using local fallback",
                data=error_data,
                request_id=request_id,
            )
        )

        return _count_tokens_local_fallback(messages, system, model_name, tools, request_id)