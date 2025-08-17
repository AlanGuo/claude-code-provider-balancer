"""
Message handling business logic for the Claude Code Provider Balancer.
Contains the core logic for processing chat completion requests.
"""

import json
import uuid
from typing import Any, Dict, Optional, Union

import httpx
import openai
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import TokenCountRequest, TokenCountResponse
from core.provider_manager import ProviderManager, Provider
from caching import (
    complete_and_cleanup_request,
    simulate_testing_delay
)
from conversion import (
    count_tokens_for_anthropic_request,
    get_anthropic_error_details_from_exc, build_anthropic_error_response
)
from utils import (
    LogRecord, LogEvent, info, error, warning, create_debug_request_info
)


def extract_detailed_error_message(error: Exception) -> tuple[str, str]:
    """
    提取详细的错误信息，返回 (error_type, detailed_message)
    """
    error_type = type(error).__name__
    
    # 处理 HTTPStatusError (httpx)
    if hasattr(error, 'response'):
        try:
            response = error.response
            
            # 尝试获取响应体文本
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content.decode('utf-8', errors='ignore')
            
            if response_text:
                # 尝试解析JSON响应
                try:
                    error_data = json.loads(response_text)
                    if isinstance(error_data, dict):
                        # 尝试从不同字段获取错误信息
                        error_msg = (
                            error_data.get('error', {}).get('message') if isinstance(error_data.get('error'), dict)
                            else error_data.get('error')
                            or error_data.get('message')
                            or error_data.get('detail')
                            or error_data.get('msg')  # 有些API使用msg字段
                        )
                        
                        if error_msg and error_msg != str(error):
                            base_error = str(error).split(': ', 1)[0] if ': ' in str(error) else str(error)
                            return error_type, f"{base_error}: {error_msg}"
                        
                except (json.JSONDecodeError, AttributeError):
                    pass
                
                # JSON解析失败或没有找到错误信息，使用原始响应文本
                if len(response_text) > 500:
                    base_error = str(error).split(': ', 1)[0] if ': ' in str(error) else str(error)
                    return error_type, f"{base_error}: {response_text[:500]}..."
                else:
                    base_error = str(error).split(': ', 1)[0] if ': ' in str(error) else str(error)
                    return error_type, f"{base_error}: {response_text}"
                    
        except Exception:
            # 获取响应信息失败，使用基本错误信息
            pass
    
    # 处理其他异常类型
    return error_type, str(error)


def _create_error_preview(response_content, max_preview_length: int = 200):
    """创建响应体的预览和完整内容"""
    
    if isinstance(response_content, list):
        # Stream response (List[str])
        full_content = "".join(response_content)
    elif isinstance(response_content, str):
        full_content = response_content
    elif isinstance(response_content, dict):
        try:
            full_content = json.dumps(response_content, ensure_ascii=False)
        except UnicodeEncodeError:
            # Handle Unicode issues in error response formatting
            full_content = json.dumps(response_content, ensure_ascii=True)
    else:
        full_content = str(response_content)
    
    # 创建预览版本（用于控制台）
    if len(full_content) > max_preview_length:
        preview = full_content[:max_preview_length] + "..."
    else:
        preview = full_content
    
    return preview, full_content


def log_provider_error(provider, error_or_reason, response_content=None, request_id: str = None, request_type: str = "unknown"):
    """
    统一记录provider错误的方法
    
    Args:
        provider: Provider对象
        error_or_reason: Exception对象或字符串错误原因
        response_content: 响应内容（可选，用于响应内容错误）
        request_id: 请求ID
        request_type: 请求类型 (streaming, non_streaming, etc.)
    """
    if isinstance(error_or_reason, Exception):
        # 处理Exception对象 - 用于连接错误等
        error_type, detailed_message = extract_detailed_error_message(error_or_reason)
        error(
            LogRecord(
                event=LogEvent.PROVIDER_REQUEST_ERROR.value,
                message=f"Provider {provider.name} {request_type} request failed: {error_type}: {detailed_message}",
                request_id=request_id,
                data={
                    "provider": provider.name,
                    "error_type": error_type,
                    "error_message": detailed_message,
                    "request_type": request_type
                }
            )
        )
    else:
        # 处理字符串错误原因 - 用于响应内容错误
        error_reason = str(error_or_reason)
        if response_content is not None:
            preview, full_content = _create_error_preview(response_content)
            error(
                LogRecord(
                    event=LogEvent.PROVIDER_REQUEST_ERROR.value,
                    message=f"Provider {provider.name} {request_type} request failed: {error_reason} - Response preview: {preview}",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "error_reason": error_reason,
                        "response_preview": preview,
                        "full_response_body": full_content,
                        "request_type": request_type
                    }
                )
            )
        else:
            error(
                LogRecord(
                    event=LogEvent.PROVIDER_REQUEST_ERROR.value,
                    message=f"Provider {provider.name} {request_type} request failed: {error_reason}",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "error_reason": error_reason,
                        "request_type": request_type
                    }
                )
            )

class MessageHandler:
    """Handles message processing requests."""
    
    def __init__(self, provider_manager: ProviderManager, settings: Any):
        self.provider_manager = provider_manager
        self.settings = settings

    def create_request_summary(self, raw_body: dict) -> str:
        """Create a concise summary of the request for logging."""
        model = raw_body.get("model", "unknown")
        stream = raw_body.get("stream", False)
        max_tokens = raw_body.get("max_tokens", 0)
        
        # Count messages
        messages_count = 0
        if "messages" in raw_body and isinstance(raw_body["messages"], list):
            messages_count = len(raw_body["messages"])
        
        # Check for tools
        tools_info = ""
        if "tools" in raw_body and isinstance(raw_body["tools"], list):
            tools_info = f", {len(raw_body['tools'])} tools"
        
        # Stream indicator - make it more prominent
        stream_info = "[STREAM]" if stream else "[NON-STREAM]"

        return f"{messages_count} msgs, max_tokens: {max_tokens}{tools_info} [{model}]{stream_info}"

    def create_no_providers_error_message(self, model: str, provider_name: str = None) -> str:
        """Create a unified error message for when no providers are available."""
        if provider_name:
            return f"Provider '{provider_name}' not found, unhealthy, or not configured for model: {model}"
        else:
            return f"All configured providers for model '{model}' are currently unable to process requests."


    async def _make_nonstreaming_http_request(self, provider: Provider, endpoint: str, data: Dict[str, Any], request_id: str, stream: bool = False, original_headers: Optional[Dict[str, str]] = None, raw_body: Optional[bytes] = None) -> Union[httpx.Response, Dict[str, Any]]:
        """Make a request to a specific provider"""
        url = self.provider_manager.get_request_url(provider, endpoint)
        headers = self.provider_manager.get_provider_headers(provider, original_headers)
        # 根据请求类型获取相应的超时配置
        http_timeouts = self.provider_manager.get_timeouts_for_request(stream)
        
        # 构建httpx超时配置
        timeout_config = httpx.Timeout(
            connect=http_timeouts['connect_timeout'],
            read=http_timeouts['read_timeout'],
            write=http_timeouts['read_timeout'],  # 使用相同的read_timeout作为write_timeout
            pool=http_timeouts['pool_timeout']
        )

        # Configure proxy if specified
        proxy_config = None
        if provider.proxy:
            proxy_config = provider.proxy

        info(
            LogRecord(
                event=LogEvent.PROVIDER_REQUEST.value,
                message=f"Making request to provider: {provider.name}",
                request_id=request_id,
                data={
                    "provider": provider.name, 
                    "type": provider.type.value,
                    "timeouts": http_timeouts
                }
            )
        )

        # Simulate testing delay if configured
        await simulate_testing_delay(data, request_id)

        # Use raw_body if provided and data hasn't been modified, otherwise serialize JSON
        headers = dict(headers) if headers else {}
        
        if raw_body is not None:
            # Use original raw body to preserve exact formatting and content-length
            request_body = raw_body
            # Preserve original content-type and content-length when using raw body
            if original_headers:
                for key, value in original_headers.items():
                    if key.lower() == 'content-type':
                        headers['content-type'] = value
                    elif key.lower() == 'content-length':
                        headers['content-length'] = value
        else:
            # Data was modified, set standard JSON content-type and calculate new content-length
            headers['content-type'] = 'application/json'
            # Data was modified, need to serialize JSON manually
            try:
                json_data = json.dumps(data, ensure_ascii=False)
                # Test if the JSON string can be safely encoded to UTF-8
                json_data.encode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Log warning for Unicode issues but continue with ASCII fallback
                warning(LogRecord(
                    event=LogEvent.REQUEST_RECEIVED.value,
                    message="Client request contains invalid Unicode characters, using ASCII encoding fallback",
                    request_id=request_id,
                    data={"provider": provider.name}
                ))
                json_data = json.dumps(data, ensure_ascii=True)
            
            request_body = json_data.encode('utf-8')
            # Set content-length manually to ensure lowercase format
            headers['content-length'] = str(len(request_body))

        async with httpx.AsyncClient(timeout=timeout_config, proxy=proxy_config) as client:
            try:
                response = await client.post(url, content=request_body, headers=headers)
                
                # Check for HTTP error status codes first (for both streaming and non-streaming)
                if response.status_code >= 400:
                    # Get response body for detailed error info
                    try:
                        error_response_body = response.json()
                    except Exception:
                        # If response is not JSON, get text content
                        error_response_body = response.text
                    
                    # Log complete error details to file (not console)
                    debug_info = create_debug_request_info(url, headers, data)
                    from utils.logging.handlers import error_file_only
                    error_file_only(
                        LogRecord(
                            event=LogEvent.PROVIDER_HTTP_ERROR_DETAILS.value,
                            message=f"Provider {provider.name} returned HTTP {response.status_code}",
                            request_id=request_id,
                            data={
                                "provider": provider.name,
                                "status_code": response.status_code,
                                "response_headers": dict(response.headers),
                                "response_body": error_response_body,
                                "request_details": debug_info
                            }
                        )
                    )
                    
                    # Create custom exception with status code for failover handling
                    from httpx import HTTPStatusError
                    request_obj = httpx.Request("POST", url)
                    
                    # Extract error message from response body if available
                    error_msg_suffix = ""
                    if error_response_body and isinstance(error_response_body, dict):
                        if "error" in error_response_body:
                            if isinstance(error_response_body["error"], str):
                                error_msg_suffix = f": {error_response_body['error']}"
                            elif isinstance(error_response_body["error"], dict) and "message" in error_response_body["error"]:
                                error_msg_suffix = f": {error_response_body['error']['message']}"
                    
                    http_error = HTTPStatusError(
                        f"HTTP {response.status_code} from provider {provider.name}{error_msg_suffix}",
                        request=request_obj,
                        response=response
                    )
                    # Add status code as attribute for error handling
                    http_error.status_code = response.status_code
                    raise http_error
                
                if stream:
                    # For streaming requests, return the response object directly
                    # HTTP errors have already been checked above
                    return response
                
                # Parse response content
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    # Handle empty or invalid JSON response
                    error_text = response.text if hasattr(response, 'text') else str(response.content)
                    
                    # If HTTP 200 with non-JSON content, return response object for further processing
                    # The ResponseHandler will handle this gracefully by returning raw content
                    if response.status_code == 200:
                        warning(LogRecord(
                            event=LogEvent.PARAMETER_UNSUPPORTED.value,
                            message=f"Provider {provider.name} returned HTTP 200 with non-JSON content in handler",
                            request_id=request_id,
                            data={
                                "provider": provider.name,
                                "status_code": response.status_code,
                                "content_preview": error_text[:200] if error_text else "empty",
                                "json_error": str(e)
                            }
                        ))
                        # Return the response object so ResponseHandler can process it
                        return response
                    else:
                        # Non-200 status with JSON error - treat as error
                        error_msg = f"Provider returned invalid JSON response. Status: {response.status_code}, Content: '{error_text[:200]}...'"
                        raise Exception(error_msg) from e
                except UnicodeDecodeError as e:
                    # Handle Unicode issues in provider response - transparently pass through
                    warning(LogRecord(
                        event=LogEvent.PROVIDER_RESPONSE.value,
                        message="Provider response contains invalid Unicode characters, returning raw text",
                        request_id=request_id,
                        data={"provider": provider.name}
                    ))
                    # Return raw text instead of parsed JSON to maintain transparency
                    return response.text
                
                # Check if response contains error even with 200 status code
                if isinstance(response_data, dict) and "error" in response_data:
                    # Create an httpx.HTTPStatusError-like exception with the error info
                    from httpx import HTTPStatusError
                    error_message = response_data.get("error", {}).get("message", "Unknown error from provider")
                    error_type = response_data.get("error", {}).get("type", "unknown_error")
                    
                    # Log detailed request information for API errors (only to file, not console)
                    debug_info = create_debug_request_info(url, headers, data)
                    error(
                        LogRecord(
                            event=LogEvent.PROVIDER_API_ERROR_DETAILS.value,
                            message=f"Provider {provider.name} returned API error: {error_message}",
                            request_id=request_id,
                            data={
                                "provider": provider.name,
                                "error_type": error_type,
                                "error_message": error_message,
                                "status_code": response.status_code,
                                "response_headers": dict(response.headers),
                                "request_details": debug_info,
                                "response_body": response_data
                            }
                        )
                    )
                    
                    # Create a mock request for the exception
                    mock_request = httpx.Request("POST", url)
                    http_error = HTTPStatusError(
                        message=f"Provider returned error: {error_message}",
                        request=mock_request,
                        response=response
                    )
                    http_error.error_type = error_type
                    raise http_error
                    
                return response_data
            
            except Exception as http_error:
                # Log the specific HTTP/connection error before it propagates up
                log_provider_error(provider, http_error, request_id=request_id, request_type="non_streaming")
                raise  # Re-raise the exception to maintain existing error handling flow

    async def _make_streaming_http_request(self, provider: Provider, endpoint: str, data: Dict[str, Any], request_id: str, original_headers: Optional[Dict[str, str]] = None, raw_body: Optional[bytes] = None):
        """Make a streaming request to a specific provider using proper streaming context"""
        url = self.provider_manager.get_request_url(provider, endpoint)
        headers = self.provider_manager.get_provider_headers(provider, original_headers)
        # Get streaming timeouts
        http_timeouts = self.provider_manager.get_timeouts_for_request(True)
        
        # Build httpx timeout configuration
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

        info(
            LogRecord(
                event=LogEvent.PROVIDER_REQUEST.value,
                message=f"Making streaming request to provider: {provider.name}",
                request_id=request_id,
                data={
                    "provider": provider.name, 
                    "type": provider.type.value,
                    "timeouts": http_timeouts
                }
            )
        )

        # Simulate testing delay if configured
        await simulate_testing_delay(data, request_id)

        # Use raw_body if provided and data hasn't been modified, otherwise serialize JSON
        headers = dict(headers) if headers else {}
        
        if raw_body is not None:
            # Use original raw body to preserve exact formatting and content-length
            request_body = raw_body
            # Preserve original content-type and content-length when using raw body
            if original_headers:
                for key, value in original_headers.items():
                    if key.lower() == 'content-type':
                        headers['content-type'] = value
                    elif key.lower() == 'content-length':
                        headers['content-length'] = value
        else:
            # Data was modified, set standard JSON content-type and calculate new content-length
            headers['content-type'] = 'application/json'
            # Data was modified, need to serialize JSON manually
            try:
                json_data = json.dumps(data, ensure_ascii=False)
                # Test if the JSON string can be safely encoded to UTF-8
                json_data.encode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Log warning for Unicode issues but continue with ASCII fallback
                warning(LogRecord(
                    event=LogEvent.REQUEST_RECEIVED.value,
                    message="Client streaming request contains invalid Unicode characters, using ASCII encoding fallback",
                    request_id=request_id,
                    data={"provider": provider.name}
                ))
                json_data = json.dumps(data, ensure_ascii=True)
            
            request_body = json_data.encode('utf-8')
            # Set content-length manually to ensure lowercase format
            headers['content-length'] = str(len(request_body))

        # Use stream context manager for true real-time streaming
        async with httpx.AsyncClient(timeout=timeout_config, proxy=proxy_config) as client:
            try:
                async with client.stream("POST", url, content=request_body, headers=headers) as response:
                    # Check for HTTP error status codes first
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        
                        # Try to parse error response body to extract specific error message
                        error_msg_suffix = ""
                        try:
                            error_response_body = json.loads(error_text.decode('utf-8'))
                            if isinstance(error_response_body, dict) and "error" in error_response_body:
                                if isinstance(error_response_body["error"], str):
                                    error_msg_suffix = f": {error_response_body['error']}"
                                elif isinstance(error_response_body["error"], dict) and "message" in error_response_body["error"]:
                                    error_msg_suffix = f": {error_response_body['error']['message']}"
                        except Exception:
                            # If parsing fails, just use the raw error text if it's short enough
                            if len(error_text) < 200:
                                error_msg_suffix = f": {error_text.decode('utf-8', errors='ignore')}"
                        
                        from httpx import HTTPStatusError
                        request_obj = httpx.Request("POST", url)
                        http_error = HTTPStatusError(
                            f"HTTP {response.status_code} from provider {provider.name}{error_msg_suffix}",
                            request=request_obj,
                            response=response
                        )
                        http_error.status_code = response.status_code
                        raise http_error
                    
                    # Return the streaming response context for real-time processing
                    yield response
            except Exception as streaming_error:
                # Log the specific streaming connection error before it propagates up
                log_provider_error(provider, streaming_error, request_id=request_id, request_type="streaming")
                raise  # Re-raise the exception to maintain existing error handling flow

    async def _make_openai_client_request(self, provider: Provider, openai_params: Dict[str, Any], request_id: str, stream: bool, original_headers: Optional[Dict[str, str]] = None) -> Any:
        """Internal method to make OpenAI client requests"""
        # Simulate testing delay if configured
        await simulate_testing_delay(openai_params, request_id)
        
        # 根据请求类型获取相应的超时配置
        openai_timeouts = self.provider_manager.get_timeouts_for_request(stream)
        
        log_event = LogEvent.PROVIDER_REQUEST
        info(
            LogRecord(
                event=log_event.value,
                message=f"Making {'stream ' if stream else ''}request to provider: {provider.name}",
                request_id=request_id,
                data={
                    "provider": provider.name, 
                    "type": provider.type.value,
                    "timeouts": openai_timeouts
                }
            )
        )
        
        # Configure proxy and timeouts for http_client
        http_client_config = {}
        if provider.proxy:
            http_client_config["proxies"] = provider.proxy
        
        # Add timeout configuration
        http_client_config["timeout"] = httpx.Timeout(
            connect=openai_timeouts['connect_timeout'],
            read=openai_timeouts['read_timeout'],
            write=openai_timeouts['read_timeout'],
            pool=openai_timeouts['pool_timeout']
        )
        
        # Use provider manager to get filtered headers, then extract only non-HTTP headers for OpenAI client
        # provider_headers = self.provider_manager.get_provider_headers(provider, original_headers)
        
        # Prepare default headers for OpenAI client
        # Extract only application-level headers, excluding HTTP transport headers and content-type
        default_headers = {
            "X-Title": self.settings.app_name,
        }
        # Add filtered headers from provider_headers, excluding HTTP transport headers
        # for key, value in provider_headers.items():
        #     if key.lower() not in ['content-type', 'content-length', 'content-encoding', 'transfer-encoding', 'host', 'authorization', 'x-api-key']:
        #         default_headers[key] = value
        
        # Create OpenAI client
        client = openai.AsyncOpenAI(
            api_key=provider.auth_value,
            base_url=provider.base_url,
            default_headers=default_headers,
            http_client=httpx.AsyncClient(**http_client_config)
        )
        
        try:
            # Make the request
            response = await client.chat.completions.create(**openai_params)
            
            # For streaming responses, attach the client to the response for later cleanup
            # The ResponseHandler will be responsible for closing the client
            if stream and hasattr(response, '__aiter__'):
                response._client = client
            else:
                # For non-streaming responses, close client immediately
                await client.close()
            
            return response
        except Exception as e:
            # Log the specific OpenAI client error before it propagates up
            request_type = "streaming" if stream else "non_streaming"
            log_provider_error(provider, e, request_id=request_id, request_type=request_type)
            # Close client on error
            try:
                await client.close()
            except Exception:
                pass
            raise

    async def make_anthropic_streaming_request(self, provider: Provider, messages_data: Dict[str, Any], request_id: str, original_headers: Optional[Dict[str, str]] = None, raw_body: Optional[bytes] = None):
        """Make a streaming request to an Anthropic-compatible provider"""
        # Always use new streaming method for real-time streaming
        # This ensures tests can detect fake streaming issues
        
        # Use new streaming method for real-time streaming
        async for response in self._make_streaming_http_request(provider, "v1/messages", messages_data, request_id, original_headers, raw_body):
            yield response

    async def make_anthropic_nonstreaming_request(self, provider: Provider, messages_data: Dict[str, Any], request_id: str, original_headers: Optional[Dict[str, str]] = None, raw_body: Optional[bytes] = None) -> Union[httpx.Response, Dict[str, Any]]:
        """Make a non-streaming request to an Anthropic-compatible provider"""
        response = await self._make_nonstreaming_http_request(provider, "v1/messages", messages_data, request_id, False, original_headers, raw_body)
        return response

    async def make_openai_streaming_request(self, provider: Provider, openai_params: Dict[str, Any], request_id: str, original_headers: Optional[Dict[str, str]] = None) -> Any:
        """Make a streaming request to an OpenAI-compatible provider"""
        return await self._make_openai_client_request(provider, openai_params, request_id, True, original_headers)

    async def make_openai_nonstreaming_request(self, provider: Provider, openai_params: Dict[str, Any], request_id: str, original_headers: Optional[Dict[str, str]] = None) -> Any:
        """Make a non-streaming request to an OpenAI-compatible provider"""
        return await self._make_openai_client_request(provider, openai_params, request_id, False, original_headers)

    async def log_and_return_error_response(
        self,
        request: Request,
        exc: Exception,
        request_id: str,
        status_code: int = 500,
        signature: str = None,
    ) -> JSONResponse:
        """Log error and return formatted error response."""
        # 统一清理signature，防止duplicate request死锁
        if signature:
            complete_and_cleanup_request(signature, exc, None, False, "error_cleanup")
        
        error_type, message, _, provider_details = get_anthropic_error_details_from_exc(exc)
        
        # Add hint about detailed logs in file only for provider-related errors
        final_message = f"Request failed: {message}"
        
        # Only add file hint for provider errors (not client validation errors)
        is_provider_error = (
            hasattr(exc, 'response') or  # HTTP errors from providers
            'provider' in str(exc).lower() or  # Provider-related errors
            any(keyword in str(exc).lower() for keyword in ['timeout', 'connection', 'network', 'api'])
        )
        
        if is_provider_error:
            if self.settings.log_file_path:
                final_message += f" (详细错误信息请查看日志文件: {self.settings.log_file_path})"
            else:
                final_message += " (详细错误信息请查看日志文件: logs/logs.jsonl)"
        
        error(
            LogRecord(
                event=LogEvent.REQUEST_FAILURE.value,
                message=final_message,
                request_id=request_id,
                data={"status_code": status_code, "error_type": error_type.value},
            ),
            exc=exc,
        )
        
        return build_anthropic_error_response(error_type, message, status_code, provider_details)

    async def count_tokens(self, request: Request) -> TokenCountResponse:
        """Estimates token count for given Anthropic messages and system prompt."""
        request_id = str(uuid.uuid4())
        
        try:
            raw_body = await request.body()
            parsed_body = json.loads(raw_body.decode('utf-8'))
            token_request = TokenCountRequest(**parsed_body)
            
            token_count = count_tokens_for_anthropic_request(
                token_request.messages,
                token_request.system,
                token_request.model,
                token_request.tools,
                request_id
            )
            
            return TokenCountResponse(input_tokens=token_count)
            
        except ValidationError as e:
            return await self.log_and_return_error_response(request, e, request_id, 400)
        except json.JSONDecodeError as e:
            return await self.log_and_return_error_response(request, e, request_id, 400)
        except Exception as e:
            return await self.log_and_return_error_response(request, e, request_id)