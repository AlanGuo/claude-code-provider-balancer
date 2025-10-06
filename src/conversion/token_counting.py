"""Token counting utilities using Anthropic official API."""

from typing import Any, Dict, List, Optional, Union

import httpx

try:
    from models import Message, SystemContent, Tool
except ImportError:
    try:
        from models.messages import Message, SystemContent
        from models.tools import Tool
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

try:
    from utils.logging import debug, LogRecord, LogEvent
except ImportError:
    try:
        from utils.logging.handlers import debug, LogRecord, LogEvent
    except ImportError:
        # Fallback implementations
        debug = lambda *args, **kwargs: None
        LogRecord = dict
        class LogEvent:
            COUNT_TOKENS_API_CALL = type('', (), {'value': 'count_tokens_api_call'})()


async def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
    provider_manager: Any = None,
) -> int:
    """
    Count tokens for an Anthropic request by calling the official API.

    This ensures 100% accuracy with Anthropic's server-side token counting.
    """
    if provider_manager is None:
        raise ValueError("provider_manager is required for accurate token counting")

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