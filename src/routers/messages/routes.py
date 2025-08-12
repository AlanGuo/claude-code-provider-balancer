"""
Core message API routes for Claude Code Provider Balancer.
Handles the main /v1/messages endpoint and token counting.
"""

import json
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .handlers import MessageHandler, log_provider_error
from models import MessagesRequest, TokenCountResponse
from core.provider_manager import ProviderManager, ProviderType
from core.provider_manager.health import should_mark_unhealthy
from core.streaming import (
    has_active_broadcaster, handle_duplicate_stream_request,
    create_broadcaster, register_broadcaster, unregister_broadcaster
)
from caching import (
    generate_request_signature, handle_duplicate_request,
    complete_and_cleanup_request, complete_and_cleanup_request_delayed
)
from conversion import (
    convert_anthropic_to_openai_messages, convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai, convert_openai_to_anthropic_response
)
from utils import LogRecord, LogEvent, info, warning, error, debug


@dataclass
class RequestContext:
    """Encapsulates all request processing context and data."""
    request_id: str
    request: Request
    raw_body: bytes
    parsed_body: Dict[str, Any]
    clean_request_body: Dict[str, Any]
    messages_request: MessagesRequest
    provider_name: Optional[str]
    signature: str
    original_headers: Dict[str, str]
    
    @property
    def is_streaming(self) -> bool:
        """Check if this is a streaming request."""
        return self.messages_request.stream or False


class ResponseHandler(ABC):
    """Base class for handling different provider response types."""
    
    @staticmethod
    def is_raw_response_from_handler(response) -> bool:
        """Check if response is a raw response object returned by handler due to HTTP 200 + non-JSON content."""
        return hasattr(response, 'text') and not isinstance(response, dict) and not hasattr(response, 'choices')
    
    @abstractmethod
    async def process_response(self, context: RequestContext, provider, target_model: str, 
                              response, request_id: str, attempt: int, message_handler, provider_manager):
        """Process the response and return appropriate FastAPI response."""
        pass


class AnthropicStreamingHandler(ResponseHandler):
    """Handle Anthropic streaming responses."""
    
    async def process_response(self, context: RequestContext, provider, target_model: str, 
                              response, request_id: str, attempt: int, message_handler, provider_manager):
        """Handle Anthropic streaming response."""
        stream_headers = {"x-provider-used": provider.name}
        collected_chunks = []
        
        # Test provider connection first before creating broadcaster
        # This allows failover if the provider fails before streaming starts
        try:
            # Get the provider stream generator
            provider_stream_generator = message_handler.make_anthropic_streaming_request(
                provider, context.clean_request_body, request_id, context.original_headers
            )
            
            # Try to get the first response object to verify connection
            first_response_obj = await provider_stream_generator.__anext__()
            
        except Exception as e:
            # Provider connection failed - let the exception propagate to trigger failover
            # Error logging is now handled in handlers layer for consistency
            raise e
        
        # Provider connection successful, now create broadcaster and streaming response
        async def stream_anthropic_response():
            """Simplified Anthropic streaming using parallel broadcaster"""
            broadcaster = None
            try:
                # Create parallel broadcaster for handling multiple clients
                broadcaster = create_broadcaster(context.request, request_id, provider.name)
                
                # Register broadcaster for duplicate request handling
                register_broadcaster(context.signature, broadcaster)
                
                # Create provider stream from response using real-time streaming
                async def provider_stream():
                    try:
                        # First yield from the already obtained response object
                        async for chunk in first_response_obj.aiter_text():
                            collected_chunks.append(chunk)
                            yield chunk
                        
                        # Then continue with the rest of the stream
                        async for response_obj in provider_stream_generator:
                            async for chunk in response_obj.aiter_text():
                                collected_chunks.append(chunk)
                                yield chunk
                    except Exception:
                        # Error will be logged by ParallelBroadcaster.stream_from_provider()
                        raise
                
                # Use broadcaster to handle parallel streaming with disconnect detection
                async for chunk in broadcaster.stream_from_provider(provider_stream()):
                    yield chunk
                    
            except Exception as e:
                error(
                    LogRecord(
                        "stream_anthropic_error",
                        f"Error in Anthropic streaming: {type(e).__name__}: {e}",
                        request_id,
                        {
                            "provider": provider.name,
                            "error": str(e)
                        }
                    )
                )
                raise
            finally:
                # Unregister broadcaster when streaming completes
                if broadcaster:
                    unregister_broadcaster(context.signature)
                
                # Check if collected chunks contain SSE error for delayed cleanup using health module
                has_sse_error = False
                if collected_chunks:
                    # Extract both raw SSE content and plain text content for error detection
                    chunks_content = "".join(collected_chunks) if isinstance(collected_chunks, list) else str(collected_chunks)
                    
                    # Extract plain text from SSE chunks for better error pattern matching
                    extracted_text_parts = []
                    
                    # Split SSE content by lines and extract text from content_block_delta events
                    for line in chunks_content.split('\n'):
                        line = line.strip()
                        if line.startswith('data: '):
                            try:
                                json_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if (json_data.get('type') == 'content_block_delta' and 
                                    'delta' in json_data and 
                                    json_data['delta'].get('type') == 'text_delta'):
                                    text_content = json_data['delta'].get('text', '')
                                    if text_content:
                                        extracted_text_parts.append(text_content)
                            except (json.JSONDecodeError, KeyError):
                                # If JSON parsing fails, continue to next line
                                continue
                    
                    # Combine extracted text parts
                    extracted_text = "".join(extracted_text_parts).strip()
                    
                    # Use extracted text for error pattern matching if available, otherwise use raw content
                    content_to_check = extracted_text if extracted_text else chunks_content
                    
                    is_unhealthy, error_reason = should_mark_unhealthy(
                        error_message=content_to_check,
                        source_type="response_body",
                        unhealthy_response_body_patterns=provider_manager.settings.get('unhealthy_response_body_patterns', [])
                    )
                    has_sse_error = is_unhealthy
                
                if has_sse_error:
                    # For SSE errors, we need to record this as an error for provider health
                    # but still use delayed cleanup for duplicate request handling
                    provider_manager.record_health_check_result(
                        provider.name, True, f"SSE error detected: {error_reason}", request_id
                    )
                    
                    # Use delayed cleanup for SSE errors to allow duplicate requests to get cached error response
                    complete_and_cleanup_request_delayed(
                        context.signature, collected_chunks, collected_chunks, True, provider.name, delay_seconds=3
                    )
                    
                    # Log SSE error completion with delayed cleanup
                    info(
                        LogRecord(
                            LogEvent.REQUEST_COMPLETED.value,
                            f"Anthropic streaming request completed with SSE error (delayed cleanup): {provider.name}",
                            request_id,
                            data={
                                "provider": provider.name,
                                "model": target_model,
                                "stream": True,
                                "provider_type": provider.type.value,
                                "chunks_count": len(collected_chunks),
                                "attempt": attempt + 1,
                                "has_sse_error": True,
                                "cleanup_type": "delayed"
                            }
                        )
                    )
                else:
                    # Normal completion - mark provider success and record health check
                    provider.mark_success()
                    provider_manager.mark_provider_success(provider.name)
                    provider_manager.record_health_check_result(
                        provider.name, False, None, request_id
                    )
                    
                    # Cache the successful response normally
                    complete_and_cleanup_request(context.signature, collected_chunks, collected_chunks, True, provider.name)
                    
                    # Log successful completion
                    info(
                        LogRecord(
                            event=LogEvent.REQUEST_COMPLETED.value,
                            message=f"Anthropic streaming request completed: {provider.name}",
                            request_id=request_id,
                            data={
                                "provider": provider.name,
                                "model": target_model,
                                "stream": True,
                                "provider_type": provider.type.value,
                                "chunks_count": len(collected_chunks),
                                "attempt": attempt + 1,
                                "has_sse_error": False,
                                "cleanup_type": "immediate"
                            }
                        )
                    )
        
        return StreamingResponse(
            stream_anthropic_response(),
            media_type="text/event-stream",
            headers=stream_headers
        )


class AnthropicNonStreamingHandler(ResponseHandler):
    """Handle Anthropic non-streaming responses."""
    
    async def process_response(self, context: RequestContext, provider, target_model: str, 
                              response, request_id: str, attempt: int, message_handler, provider_manager):
        """Handle Anthropic non-streaming response."""
        # Check if handler returned a raw response object due to HTTP 200 + non-JSON content
        if self.is_raw_response_from_handler(response):
            # Handler returned raw response object - treat as successful and return raw content
            raw_content = response.text
            
            # Cache the response
            complete_and_cleanup_request(context.signature, raw_content, raw_content, False, provider.name)
            
            # Mark provider success for sticky routing and failure count reset
            provider.mark_success()
            provider_manager.mark_provider_success(provider.name)
            # Record successful health check result
            provider_manager.record_health_check_result(
                provider.name, False, None, request_id
            )
            
            # Log request completion
            info(
                LogRecord(
                    event=LogEvent.REQUEST_COMPLETED.value,
                    message=f"Non-streaming request completed with non-JSON content: {provider.name}",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "model": target_model,
                        "stream": False,
                        "provider_type": provider.type.value,
                        "attempt": attempt + 1,
                        "content_type": "non-json"
                    }
                )
            )
            
            # Return as plain text response
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=raw_content, status_code=200)
        
        # Normal case: handler returned parsed JSON content
        if hasattr(response, 'json'):
            response_content = response.json()
        elif isinstance(response, dict):
            response_content = response
        else:
            # Convert non-serializable response objects to dict
            response_content = {"error": {"type": "serialization_error", "message": "Unable to serialize response"}}
        
        # Check provider health based on response content
        # Get HTTP status code from response
        response_status_code = None
        if hasattr(response, 'status_code'):
            response_status_code = response.status_code
        
        # 使用简化的健康检查
        is_error_detected, error_reason = should_mark_unhealthy(
            http_status_code=response_status_code,
            error_message=str(response_content),
            source_type="response_body",
            unhealthy_http_codes=provider_manager.settings.get('unhealthy_http_codes', []),
            unhealthy_response_body_patterns=provider_manager.settings.get('unhealthy_response_body_patterns', [])
        )
        
        # 使用ProviderManager的错误计数机制
        provider_marked_unhealthy = provider_manager.record_health_check_result(
            provider.name, is_error_detected, error_reason, request_id
        )
        
        if provider_marked_unhealthy:
            warning(
                LogRecord(
                    LogEvent.PROVIDER_UNHEALTHY_NON_STREAM.value,
                    f"Provider {provider.name} marked unhealthy after reaching error threshold",
                    request_id,
                    {
                        "provider": provider.name,
                        "error_reason": error_reason,
                        "can_failover": True,
                        "action": "marked_unhealthy_and_failover"
                    }
                )
            )
            # Re-raise exception to trigger failover
            raise Exception(f"Provider {provider.name} is unhealthy: {error_reason}")
        
        if is_error_detected:
            # 如果检测到错误，记录 PROVIDER_REQUEST_ERROR 日志
            log_provider_error(provider, error_reason, response_content, request_id, "non_streaming")
        
        # Cache the response
        complete_and_cleanup_request(context.signature, response_content, response_content, False, provider.name)
        
        # Mark provider success for sticky routing and failure count reset
        provider.mark_success()
        provider_manager.mark_provider_success(provider.name)
        # 记录成功的健康检查结果
        provider_manager.record_health_check_result(
            provider.name, False, None, request_id
        )
        
        # Log successful completion
        info(
            LogRecord(
                event=LogEvent.REQUEST_COMPLETED.value,
                message=f"Non-streaming request completed: {provider.name}",
                request_id=request_id,
                data={
                    "provider": provider.name,
                    "model": target_model,
                    "stream": False,
                    "provider_type": provider.type.value,
                    "attempt": attempt + 1,
                }
            )
        )
        
        return JSONResponse(content=response_content)


class OpenAIStreamingHandler(ResponseHandler):
    """Handle OpenAI streaming responses."""
    
    async def process_response(self, context: RequestContext, provider, target_model: str, 
                              response, request_id: str, attempt: int, message_handler, provider_manager):
        """Handle OpenAI streaming response."""
        stream_headers = {"x-provider-used": provider.name}
        
        if hasattr(response, '__aiter__'):
            # Response is an AsyncStream object, collect chunks for caching while streaming
            collected_chunks = []
            
            async def stream_openai_response():
                """Handle OpenAI streaming response and convert to Anthropic format"""
                broadcaster = None
                try:
                    # Create parallel broadcaster for handling multiple clients
                    broadcaster = create_broadcaster(context.request, request_id, provider.name)
                    
                    # Register broadcaster for duplicate request handling
                    register_broadcaster(context.signature, broadcaster)
                    
                    # Create provider stream from OpenAI AsyncStream
                    async def provider_stream():
                        try:
                            async for chunk in response:
                                # Convert OpenAI chunk to Anthropic SSE format
                                if hasattr(chunk, 'choices') and chunk.choices:
                                    choice = chunk.choices[0]
                                    
                                    if hasattr(choice, 'delta') and choice.delta:
                                        delta = choice.delta
                                        if hasattr(delta, 'content') and delta.content:
                                            # Create Anthropic-style text delta event
                                            anthropic_chunk = {
                                                "type": "content_block_delta",
                                                "index": 0,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": delta.content
                                                }
                                            }
                                            sse_data = f"data: {json.dumps(anthropic_chunk)}\n\n"
                                            collected_chunks.append(sse_data)
                                            yield sse_data
                                        elif hasattr(choice, 'finish_reason') and choice.finish_reason:
                                            # Create Anthropic-style message stop event
                                            stop_chunk = {
                                                "type": "message_stop"
                                            }
                                            sse_data = f"data: {json.dumps(stop_chunk)}\n\n"
                                            collected_chunks.append(sse_data)
                                            yield sse_data
                        except Exception as e:
                            error(
                                LogRecord(
                                    "openai_stream_error",
                                    f"Error in OpenAI stream: {type(e).__name__}: {e}",
                                    request_id,
                                    {
                                        "provider": provider.name,
                                        "error": str(e)
                                    }
                                )
                            )
                            raise
                    
                    # Use broadcaster to handle parallel streaming with disconnect detection
                    async for chunk in broadcaster.stream_from_provider(provider_stream()):
                        yield chunk
                        
                except Exception as e:
                    error(
                        LogRecord(
                            "stream_openai_error",
                            f"Error in OpenAI streaming: {type(e).__name__}: {e}",
                            request_id,
                            {
                                "provider": provider.name,
                                "error": str(e)
                            }
                        )
                    )
                    raise
                finally:
                    # Close OpenAI client if it's attached to the response
                    if hasattr(response, '_client') and response._client:
                        try:
                            await response._client.close()
                        except Exception:
                            pass  # Ignore errors when closing client
                    
                    # Unregister broadcaster when streaming completes
                    if broadcaster:
                        unregister_broadcaster(context.signature)
                    
                    # Complete the request with collected chunks
                    complete_and_cleanup_request(context.signature, collected_chunks, collected_chunks, True, provider.name)
                    
                    # Mark provider success for sticky routing and failure count reset
                    provider.mark_success()
                    provider_manager.mark_provider_success(provider.name)
                    # 记录成功的健康检查结果
                    provider_manager.record_health_check_result(
                        provider.name, False, None, request_id
                    )
                    
                    # Log request completion
                    info(
                        LogRecord(
                            event=LogEvent.REQUEST_COMPLETED.value,
                            message=f"OpenAI streaming request completed: {provider.name}",
                            request_id=request_id,
                            data={
                                "provider": provider.name,
                                "model": target_model,
                                "stream": True,
                                "provider_type": "openai",
                                "chunks_count": len(collected_chunks),
                                "attempt": attempt + 1,
                            }
                        )
                    )
            
            return StreamingResponse(
                stream_openai_response(),
                media_type="text/event-stream",
                headers=stream_headers
            )
        else:
            # If response is not streamable, convert to streaming format
            response_data = response.json() if hasattr(response, 'json') else response
            collected_chunks = [f"data: {json.dumps(response_data)}\n\n"]
            
            async def convert_to_stream():
                yield f"data: {json.dumps(response_data)}\n\n"
            
            complete_and_cleanup_request(context.signature, response_data, collected_chunks, True, provider.name)
            
            return StreamingResponse(
                convert_to_stream(),
                media_type="text/event-stream",
                headers=stream_headers
            )


class OpenAINonStreamingHandler(ResponseHandler):
    """Handle OpenAI non-streaming responses."""
    
    async def process_response(self, context: RequestContext, provider, target_model: str, 
                              response, request_id: str, attempt: int, message_handler, provider_manager):
        """Handle OpenAI non-streaming response."""
        # Check if handler returned a raw response object due to HTTP 200 + non-JSON content
        if self.is_raw_response_from_handler(response):
            # Handler returned raw response object - treat as successful and return raw content
            raw_content = response.text
            
            # Cache the response
            complete_and_cleanup_request(context.signature, raw_content, raw_content, False, provider.name)
            
            # Mark provider success for sticky routing and failure count reset
            provider.mark_success()
            provider_manager.mark_provider_success(provider.name)
            # Record successful health check result
            provider_manager.record_health_check_result(
                provider.name, False, None, request_id
            )
            
            # Log request completion
            info(
                LogRecord(
                    event=LogEvent.REQUEST_COMPLETED.value,
                    message=f"OpenAI non-streaming request completed with conversion error, returning raw content: {provider.name}",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "model": target_model,
                        "stream": False,
                        "provider_type": provider.type.value,
                        "attempt": attempt + 1,
                        "content_type": "conversion-failed"
                    }
                )
            )
            
            # Return as plain text response
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=raw_content, status_code=200)
        
        # Normal OpenAI response conversion
        anthropic_response = convert_openai_to_anthropic_response(response, request_id)
        # Convert Pydantic model to dict for JSON serialization
        response_content = anthropic_response.model_dump()
        
        # Check provider health based on response content
        # Get HTTP status code from response
        response_status_code = None
        if hasattr(response, 'status_code'):
            response_status_code = response.status_code
        
        # 使用简化的健康检查
        is_error_detected, error_reason = should_mark_unhealthy(
            http_status_code=response_status_code,
            error_message=str(response_content),
            source_type="response_body",
            unhealthy_http_codes=provider_manager.settings.get('unhealthy_http_codes', []),
            unhealthy_response_body_patterns=provider_manager.settings.get('unhealthy_response_body_patterns', [])
        )
        
        # 使用ProviderManager的错误计数机制
        provider_marked_unhealthy = provider_manager.record_health_check_result(
            provider.name, is_error_detected, error_reason, request_id
        )
        
        if provider_marked_unhealthy:
            warning(
                LogRecord(
                    LogEvent.PROVIDER_UNHEALTHY_NON_STREAM.value,
                    f"Provider {provider.name} marked unhealthy after reaching error threshold",
                    request_id,
                    {
                        "provider": provider.name,
                        "error_reason": error_reason,
                        "can_failover": True,
                        "action": "marked_unhealthy_and_failover"
                    }
                )
            )
            # Re-raise exception to trigger failover
            raise Exception(f"Provider {provider.name} is unhealthy: {error_reason}")
        
        if is_error_detected:
            # 如果检测到错误，记录 PROVIDER_REQUEST_ERROR 日志
            log_provider_error(provider, error_reason, response_content, request_id, "non_streaming")
        
        # Cache the response
        complete_and_cleanup_request(context.signature, response_content, response_content, False, provider.name)
        
        # Mark provider success for sticky routing and failure count reset
        provider.mark_success()
        provider_manager.mark_provider_success(provider.name)
        # 记录成功的健康检查结果
        provider_manager.record_health_check_result(
            provider.name, False, None, request_id
        )
        
        # Log successful completion
        info(
            LogRecord(
                event=LogEvent.REQUEST_COMPLETED.value,
                message=f"Non-streaming request completed: {provider.name}",
                request_id=request_id,
                data={
                    "provider": provider.name,
                    "model": target_model,
                    "stream": False,
                    "provider_type": provider.type.value,
                    "attempt": attempt + 1,
                }
            )
        )
        
        return JSONResponse(content=response_content)


def get_response_handler(provider_type: ProviderType, is_streaming: bool) -> ResponseHandler:
    """Factory method to get the appropriate response handler."""
    if provider_type == ProviderType.ANTHROPIC:
        return AnthropicStreamingHandler() if is_streaming else AnthropicNonStreamingHandler()
    elif provider_type == ProviderType.OPENAI:
        return OpenAIStreamingHandler() if is_streaming else OpenAINonStreamingHandler()
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")




def create_messages_router(provider_manager: ProviderManager, settings: Any) -> APIRouter:
    """Create messages router with dependencies."""
    router = APIRouter(prefix="/v1", tags=["API"])
    message_handler = MessageHandler(provider_manager, settings)

    async def _preprocess_request(request: Request, request_id: str) -> RequestContext:
        """Extract and validate request data, create context object."""
        # Get request body for logging and caching
        raw_body = await request.body()
        parsed_body = json.loads(raw_body.decode('utf-8', errors='ignore'))
        
        # Extract provider parameter separately before validation
        provider_name = parsed_body.pop("provider", None)
        
        # Generate request signature for deduplication (without provider)
        signature = generate_request_signature(parsed_body)
        
        # Validate the remaining fields with MessagesRequest
        messages_request = MessagesRequest(**parsed_body)
        
        # Add provider back to parsed_body for logging and other uses
        if provider_name:
            parsed_body["provider"] = provider_name
        
        # Log request received
        info(
            LogRecord(
                event=LogEvent.REQUEST_RECEIVED.value,
                message=message_handler.create_request_summary(parsed_body),
                request_id=request_id,
                data={
                    "model": messages_request.model,
                    "stream": messages_request.stream,
                    "provider": provider_name,
                },
            )
        )
        
        # Create clean request body without balancer-specific fields for provider requests
        clean_request_body = {k: v for k, v in parsed_body.items() if k not in ['provider']}
        original_headers = dict(request.headers)
        return RequestContext(
            request_id=request_id,
            request=request,
            raw_body=raw_body,
            parsed_body=parsed_body,
            clean_request_body=clean_request_body,
            messages_request=messages_request,
            provider_name=provider_name,
            signature=signature,
            original_headers=original_headers
        )

    async def _handle_duplicate_requests(context: RequestContext, request_id: str) -> Optional[StreamingResponse]:
        """Handle duplicate request detection and processing."""
        # For streaming requests, first check if there's an active broadcaster for this signature
        if context.is_streaming and has_active_broadcaster(context.signature):
            try:
                # Try to connect to existing broadcaster for duplicate stream request
                stream_generator = handle_duplicate_stream_request(context.signature, context.request, request_id)
                return StreamingResponse(
                    stream_generator,
                    media_type="text/event-stream",
                    headers={"x-provider-used": "broadcaster-duplicate"}
                )
            except Exception as e:
                # Fallback if broadcaster connection fails
                debug(
                    LogRecord(
                        "broadcaster_connection_failed",
                        f"Failed to connect to active broadcaster: {str(e)}",
                        request_id,
                        {
                            "signature": context.signature[:16] + "...",
                            "error": str(e)
                        }
                    )
                )
        
        # Check for cached duplicate requests
        duplicate_result = await handle_duplicate_request(
            context.signature, request_id, context.is_streaming, context.clean_request_body
        )
        return duplicate_result

    async def _select_provider_options(context: RequestContext, request_id: str) -> list:
        """Select available provider options for failover."""
        # Select all available provider options for failover
        provider_options = provider_manager.select_model_and_provider_options(
            context.messages_request.model, context.provider_name
        )
        
        if not provider_options:
            error_msg = message_handler.create_no_providers_error_message(
                context.messages_request.model, context.provider_name
            )
            raise Exception(error_msg)
        
        # Log available options
        info(
            LogRecord(
                event=LogEvent.REQUEST_START.value,
                message=f"Processing request with {len(provider_options)} provider option(s)",
                request_id=request_id,
                data={
                    "client_model": context.messages_request.model,
                    "available_options": len(provider_options),
                    "primary_provider": provider_options[0][1].name,
                    "stream": context.is_streaming,
                },
            )
        )
        
        return provider_options

    async def _execute_provider_request(context: RequestContext, provider, target_model: str, request_id: str):
        """Execute request for a single provider."""
        if provider.type == ProviderType.ANTHROPIC:
            # For Anthropic providers, use the appropriate method based on streaming
            if context.is_streaming:
                # Return the async generator for streaming
                return message_handler.make_anthropic_streaming_request(
                    provider, context.clean_request_body, request_id, context.original_headers
                )
            else:
                return await message_handler.make_anthropic_nonstreaming_request(
                    provider, context.clean_request_body, request_id, context.original_headers
                )
        elif provider.type == ProviderType.OPENAI:
            # Convert to OpenAI format first
            openai_messages = convert_anthropic_to_openai_messages(
                context.messages_request.messages, context.messages_request.system, request_id
            )
            openai_tools = convert_anthropic_tools_to_openai(context.messages_request.tools)
            openai_tool_choice = convert_anthropic_tool_choice_to_openai(
                context.messages_request.tool_choice, request_id
            )
            
            openai_params = {
                "model": target_model,
                "messages": openai_messages,
                "max_tokens": context.messages_request.max_tokens,
                "stream": context.is_streaming,
            }
            
            if context.messages_request.temperature is not None:
                openai_params["temperature"] = context.messages_request.temperature
            if context.messages_request.top_p is not None:
                openai_params["top_p"] = context.messages_request.top_p
            if context.messages_request.stop_sequences:
                openai_params["stop"] = context.messages_request.stop_sequences
            if openai_tools:
                openai_params["tools"] = openai_tools
            if openai_tool_choice:
                openai_params["tool_choice"] = openai_tool_choice
            
            if context.is_streaming:
                return await message_handler.make_openai_streaming_request(
                    provider, openai_params, request_id, context.original_headers
                )
            else:
                return await message_handler.make_openai_nonstreaming_request(
                    provider, openai_params, request_id, context.original_headers
                )
        else:
            raise ValueError(f"Unsupported provider type: {provider.type}")

    @router.post("/messages", response_model=None, status_code=200)
    async def create_message_proxy(request: Request) -> JSONResponse:
        """Proxy endpoint for Anthropic Messages API."""
        request_id = str(uuid.uuid4())
        
        try:
            # Check and reset timed-out provider errors before processing request
            provider_manager.check_and_reset_timeout_errors()
            
            # Preprocess request and create context
            context = await _preprocess_request(request, request_id)
            
            # Handle duplicate requests
            duplicate_result = await _handle_duplicate_requests(context, request_id)
            if duplicate_result is not None:
                return duplicate_result
            
            # Select provider options for failover
            try:
                provider_options = await _select_provider_options(context, request_id)
            except Exception as e:
                return await message_handler.log_and_return_error_response(
                    request, e, request_id, 404, context.signature
                )
            
            # Try providers in order until one succeeds
            max_attempts = len(provider_options)
            last_exception = None
            
            for attempt in range(max_attempts):
                target_model, current_provider = provider_options[attempt]
                
                try:
                    # Execute request for current provider
                    response = await _execute_provider_request(context, current_provider, target_model, request_id)
                    
                    # Get appropriate response handler using strategy pattern
                    handler = get_response_handler(current_provider.type, context.is_streaming)
                    
                    # Process response using the selected handler
                    return await handler.process_response(
                        context, current_provider, target_model, response, 
                        request_id, attempt, message_handler, provider_manager
                    )
                except Exception as e:
                    last_exception = e
                    
                    # Get HTTP status code if available
                    http_status_code = getattr(e, 'status_code', None) or (
                        getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
                    )
                    
                    # Special handling for 401 Unauthorized and 403 Forbidden with Claude Code Official
                    if http_status_code in [401, 403] and current_provider.name == "Claude Code Official":
                        # Handle OAuth authorization required
                        provider_manager.handle_oauth_authorization_required(current_provider, http_status_code)
                    
                    # Use provider_manager to determine error handling strategy
                    error_reason, should_record_error, can_failover = provider_manager.get_error_handling_decision(
                        e, http_status_code, context.messages_request.stream
                    )
                    
                    # Mark current provider as failed if unhealthy threshold is reached
                    provider_marked_unhealthy = False
                    # Use error counting mechanism before marking as failed
                    provider_marked_unhealthy = provider_manager.record_health_check_result(
                        current_provider.name, should_record_error, error_reason, request_id
                    )
                    # Only mark as failed if threshold is reached
                    if provider_marked_unhealthy:
                        current_provider.mark_failure()
                    # If threshold not reached, don't mark as failed - just record the error
                    
                    # Only attempt failover if provider was marked as unhealthy
                    if not provider_marked_unhealthy:
                        # Provider not marked unhealthy, return error immediately (no failover needed)
                        provider_manager.mark_provider_used(current_provider.name)
                        
                        # Get current error status for logging
                        error_status = provider_manager.get_provider_error_status(current_provider.name)
                        error_count = error_status.get("error_count", 0)
                        threshold = error_status.get("threshold", 2)
                        
                        debug(
                            LogRecord(
                                event=LogEvent.PROVIDER_ERROR_BELOW_THRESHOLD.value,
                                message=f"Provider not marked unhealthy: count={error_count}/{threshold}, returning error to client",
                                request_id=request_id,
                                data={
                                    "provider": current_provider.name,
                                    "error_reason": error_reason,
                                    "http_status_code": http_status_code,
                                    "provider_marked_unhealthy": provider_marked_unhealthy,
                                    "error_count": error_count,
                                    "threshold": threshold
                                }
                            )
                        )
                        # 从异常中提取HTTP状态码，保持原始错误状态码
                        http_status_code_from_exc = getattr(e, 'status_code', None) or (
                            getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
                        )
                        status_code = http_status_code_from_exc if http_status_code_from_exc else 500
                        return await message_handler.log_and_return_error_response(request, e, request_id, status_code, context.signature)
                    
                    # Provider was marked unhealthy, now check if we can failover
                    if not can_failover:
                        # Provider is unhealthy but we cannot failover (e.g., streaming response headers already sent)
                        info(
                            LogRecord(
                                event=LogEvent.PROVIDER_UNHEALTHY_NO_FAILOVER.value,
                                message="Provider marked unhealthy but cannot failover for this request type, returning error to client",
                                request_id=request_id,
                                data={
                                    "provider": current_provider.name,
                                    "error_reason": error_reason,
                                    "can_failover": can_failover,
                                    "is_streaming": context.messages_request.stream,
                                    "provider_marked_unhealthy": provider_marked_unhealthy
                                }
                            )
                        )
                        # 从异常中提取HTTP状态码，保持原始错误状态码
                        http_status_code_from_exc = getattr(e, 'status_code', None) or (
                            getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
                        )
                        status_code = http_status_code_from_exc if http_status_code_from_exc else 500
                        return await message_handler.log_and_return_error_response(request, e, request_id, status_code, context.signature)
                    
                    # If we have more providers to try, continue to next iteration
                    if attempt < max_attempts - 1:
                        next_target_model, next_provider = provider_options[attempt + 1]
                        info(
                            LogRecord(
                                event=LogEvent.PROVIDER_FALLBACK.value,
                                message=f"Falling back to provider: {next_provider.name} with model: {next_target_model}",
                                request_id=request_id,
                                data={
                                    "failed_provider": current_provider.name,
                                    "failed_model": target_model,
                                    "fallback_provider": next_provider.name,
                                    "fallback_model": next_target_model,
                                    "attempt": attempt + 2,
                                    "total_attempts": max_attempts
                                }
                            )
                        )
                    # If this is the last attempt, the loop will end and we'll return the error
            
            # All providers failed, return ALL_PROVIDERS_FAILED
            error(
                LogRecord(
                    event=LogEvent.ALL_PROVIDERS_FAILED.value,
                    message=f"All {max_attempts} provider(s) failed for model: {context.messages_request.model}",
                    request_id=request_id,
                    data={
                        "model": context.messages_request.model,
                        "total_attempts": max_attempts,
                        "providers_tried": [opt[1].name for opt in provider_options]
                    }
                ),
                exc=last_exception
            )
            
            # Create a generic error message for the client that doesn't expose provider details
            client_error_message = message_handler.create_no_providers_error_message(context.messages_request.model)
            client_error = Exception(client_error_message)
            
            # 当所有providers都不可用时，返回503 Service Unavailable
            return await message_handler.log_and_return_error_response(request, client_error, request_id, 503, context.signature)
                
        except ValidationError as e:
            # ValidationError发生在生成signature之前，无需cleanup
            return await message_handler.log_and_return_error_response(request, e, request_id, 400)
        except json.JSONDecodeError as e:
            # JSONDecodeError发生在生成signature之前，无需cleanup
            return await message_handler.log_and_return_error_response(request, e, request_id, 400)
        except Exception as e:
            # 通用异常可能在任何阶段发生，尝试cleanup（如果signature不存在会被安全忽略）
            # 尝试从异常中提取HTTP状态码，如果没有则默认为500
            http_status_code = getattr(e, 'status_code', None) or (
                getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
            )
            status_code = http_status_code if http_status_code else 500
            return await message_handler.log_and_return_error_response(request, e, request_id, status_code, locals().get('signature'))

    @router.post("/messages/count_tokens", response_model=TokenCountResponse)
    async def count_tokens_endpoint(request: Request) -> TokenCountResponse:
        """Estimates token count for given Anthropic messages and system prompt."""
        return await message_handler.count_tokens(request)

    return router