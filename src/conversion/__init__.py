"""Conversion utilities for translating between Anthropic and OpenAI API formats."""

from .token_counting import (
    count_tokens_for_anthropic_request
)

from .anthropic_to_openai import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai
)

from .openai_to_anthropic import (
    convert_openai_to_anthropic_response,
    handle_anthropic_streaming_response_from_openai_stream
)

from .error_handling import (
    get_anthropic_error_details_from_exc,
    build_anthropic_error_response
)

from .helpers import (
    serialize_tool_result_content_for_openai
)

__all__ = [
    # Token counting
    "count_tokens_for_anthropic_request",
    
    # Anthropic to OpenAI
    "convert_anthropic_to_openai_messages",
    "convert_anthropic_tools_to_openai", 
    "convert_anthropic_tool_choice_to_openai",
    
    # OpenAI to Anthropic
    "convert_openai_to_anthropic_response",
    "handle_anthropic_streaming_response_from_openai_stream",
    
    # Error handling
    "get_anthropic_error_details_from_exc",
    "build_anthropic_error_response",
    
    # Helpers
    "serialize_tool_result_content_for_openai"
]