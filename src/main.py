"""
Single-file FastAPI application to proxy Anthropic API requests to an OpenAI-compatible API (e.g., OpenRouter).
Handles request/response conversion, streaming, and dynamic model selection.
"""

import asyncio
import dataclasses
import enum
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from logging.config import dictConfig
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, List,
                    Literal, Optional, Tuple, Union, cast)

import fastapi
import openai
import tiktoken
import uvicorn
import httpx
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionToolParam)
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from provider_manager import ProviderManager, Provider, ProviderType

load_dotenv()


def _create_body_summary(raw_body: dict) -> dict:
    """Create a summary of the request body for logging purposes."""
    summary = {}

    # Include basic request info
    if "model" in raw_body:
        summary["model"] = raw_body["model"]

    if "max_tokens" in raw_body:
        summary["max_tokens"] = raw_body["max_tokens"]

    if "stream" in raw_body:
        summary["stream"] = raw_body["stream"]

    # Summarize messages
    if "messages" in raw_body and isinstance(raw_body["messages"], list):
        messages = raw_body["messages"]
        summary["messages_count"] = len(messages)

        # Include role info and content length for each message
        messages_summary = []
        for msg in messages:
            msg_summary = {}
            if "role" in msg:
                msg_summary["role"] = msg["role"]
            if "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    msg_summary["content_length"] = len(content)
                    msg_summary["content_preview"] = content[:100] + "..." if len(content) > 100 else content
                elif isinstance(content, list):
                    msg_summary["content_blocks"] = len(content)
                    msg_summary["content_types"] = [block.get("type", "unknown") for block in content if isinstance(block, dict)]
                else:
                    msg_summary["content_type"] = type(content).__name__
            messages_summary.append(msg_summary)

        summary["messages"] = messages_summary

    # Include other important fields
    for key in ["temperature", "top_p", "top_k", "stop_sequences", "tools", "system"]:
        if key in raw_body:
            if key == "tools" and isinstance(raw_body[key], list):
                summary[key + "_count"] = len(raw_body[key])
            elif key == "system" and isinstance(raw_body[key], str):
                system_content = raw_body[key]
                summary[key + "_length"] = len(system_content)
                summary[key + "_preview"] = system_content[:100] + "..." if len(system_content) > 100 else system_content
            else:
                summary[key] = raw_body[key]

    return summary


class Settings(BaseSettings):
    """Application settings loaded from environment variables and provider config."""

    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")

    # Optional environment overrides
    log_level: str = "INFO"
    log_file_path: str = ""
    log_color: bool = True
    providers_config_path: str = "providers.yaml"
    referrer_url: str = "http://localhost:8082/claude_proxy"
    reload: bool = False

    # These will be loaded from provider config
    host: str = "127.0.0.1"
    port: int = 8080
    app_name: str = "Claude Code Provider Balancer"
    app_version: str = "0.3.0"


# Initialize provider manager and settings
try:
    provider_manager = ProviderManager()
    settings = Settings()

    # Override with provider config settings if available
    provider_settings = provider_manager.settings
    if provider_settings:
        settings.host = provider_settings.get('host', settings.host)
        settings.port = provider_settings.get('port', settings.port)
        settings.log_level = provider_settings.get('log_level', settings.log_level)
        settings.log_color = provider_settings.get('log_color', settings.log_color)
        settings.app_name = provider_settings.get('app_name', settings.app_name)
        settings.app_version = provider_settings.get('app_version', settings.app_version)
except Exception as e:
    # Fallback to basic settings if provider config fails
    print(f"Warning: Failed to load provider configuration: {e}")
    print("Using basic settings...")
    provider_manager = None
    settings = Settings()


_console = Console()
_error_console = Console(stderr=True, style="bold red")

# Recursion protection for exception handling
_exception_handler_lock = threading.RLock()
_exception_handler_depth = threading.local()

# 请求去重状态管理
_pending_requests: Dict[str, Tuple[asyncio.Future, str]] = {}
_request_cleanup_lock = threading.RLock()

def _generate_request_signature(data: Dict[str, Any]) -> str:
    """为请求生成唯一签名用于去重"""
    
    # 检查是否为流式请求
    is_stream = data.get("stream", False)
    
    # 流式请求禁用去重，因为StreamingResponse不能被重复消费
    if is_stream:
        import uuid
        unique_stream_signature = f"stream_{uuid.uuid4().hex}"
        debug(
            LogRecord(
                "streaming_dedup_disabled",
                f"Streaming request deduplication disabled, using unique signature",
                None,
                {"signature": unique_stream_signature[:16] + "..."},
            )
        )
        return unique_stream_signature
    
    # 非流式请求正常去重
    signature_data = {
        "model": data.get("model", ""),
        "messages": data.get("messages", []),
        "system": data.get("system", ""),
        "tools": data.get("tools", []),
        "temperature": data.get("temperature", 0),
        # 注意：不包含 stream 字段，让流式和非流式请求共享去重
    }
    
    # 根据配置决定是否包含 max_tokens 字段
    include_max_tokens = provider_manager.settings.get("deduplication", {}).get("include_max_tokens_in_signature", True) if provider_manager else True
    if include_max_tokens:
        signature_data["max_tokens"] = data.get("max_tokens", 0)

    # 将数据转换为可哈希的字符串
    signature_str = json.dumps(signature_data, sort_keys=True, separators=(',', ':'))

    # 生成 SHA256 哈希
    signature_hash = hashlib.sha256(signature_str.encode('utf-8')).hexdigest()

    return signature_hash

def _cleanup_completed_request(signature: str):
    """清理已完成的请求"""
    with _request_cleanup_lock:
        if signature in _pending_requests:
            del _pending_requests[signature]


def _complete_and_cleanup_request(signature: str, result: Any):
    """完成请求并清理去重状态"""
    if signature:
        try:
            # 设置 Future 结果并清理
            with _request_cleanup_lock:
                if signature in _pending_requests:
                    future, original_request_id = _pending_requests[signature]
                    if not future.done():
                        future.set_result(result)
                    del _pending_requests[signature]
                    
                    # 根据结果类型确定清理原因
                    if isinstance(result, Exception):
                        cleanup_reason = "request_failed"
                        message = f"Request failed and cleaned up (duplicate request cleared): {str(result)}"
                    elif isinstance(result, str):
                        if "streaming" in result:
                            cleanup_reason = "streaming_completed"
                            message = f"Streaming request completed and cleaned up (duplicate request cleared)"
                        else:
                            cleanup_reason = "request_completed"
                            message = f"Request completed and cleaned up (duplicate request cleared)"
                    else:
                        cleanup_reason = "request_completed"
                        message = f"Request completed and cleaned up (duplicate request cleared)"
                    
                    info(
                        LogRecord(
                            "request_cleanup",
                            message,
                            original_request_id,
                            {
                                "request_signature": signature[:16] + "...",
                                "result_type": type(result).__name__,
                                "pending_requests_count": len(_pending_requests),
                                "cleanup_reason": cleanup_reason
                            },
                        )
                    )
                else:
                    # 这种情况是正常的，因为非重复请求不会在pending_requests中
                    debug(
                        LogRecord(
                            "request_cleanup_skip",
                            f"No pending request found for signature (non-duplicate request)",
                            None,
                            {"signature": signature[:16] + "..."},
                        )
                    )
        except Exception as e:
            error(
                LogRecord(
                    "request_cleanup_error",
                    f"Error during request cleanup: {str(e)}",
                    None,
                    {"signature": signature[:16] + "..."},
                )
            )

async def _handle_duplicate_request(signature: str, request_id: str) -> Optional[Any]:
    """处理重复请求，如果是重复请求则等待原请求完成"""
    future_to_wait = None
    original_request_id = None

    with _request_cleanup_lock:
        if signature in _pending_requests:
            # 这是重复请求，获取 Future 但不在锁内等待
            future_to_wait, original_request_id = _pending_requests[signature]
            info(
                LogRecord(
                    LogEvent.REQUEST_RECEIVED.value,
                    "Duplicate request detected, waiting for original request to complete",
                    request_id,
                    {"original_request_id": original_request_id, "signature": signature[:16] + "..."},
                )
            )
        else:
            # 这是新请求，创建 Future 并记录
            future = asyncio.Future()
            _pending_requests[signature] = (future, request_id)
            return None  # 表示这是新请求，继续处理

    # 在锁外等待原请求完成
    if future_to_wait:
        try:
            result = await future_to_wait
            info(
                LogRecord(
                    LogEvent.REQUEST_COMPLETED.value,
                    "Duplicate request completed via original request",
                    request_id,
                    {"original_request_id": original_request_id, "signature": signature[:16] + "..."},
                )
            )
            return result
        except Exception as e:
            # 原请求失败，重复请求也应该收到相同的错误
            info(
                LogRecord(
                    LogEvent.REQUEST_FAILED.value,
                    "Duplicate request failed via original request",
                    request_id,
                    {"original_request_id": original_request_id, "signature": signature[:16] + "...", "error": str(e)},
                )
            )
            raise e

    return None


@dataclasses.dataclass
class LogError:
    name: str
    message: str
    stack_trace: Optional[str] = None
    args: Optional[Tuple[Any, ...]] = None


@dataclasses.dataclass
class LogRecord:
    event: str
    message: str
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[LogError] = None


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color support based on log level."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Get the base JSON output
        log_dict = self._get_log_dict(record)

        # Check if we should use colors (only for TTY output and when enabled)
        # Use globals() to get the current settings value at runtime
        current_settings = globals().get('settings')
        use_colors = (
            current_settings is not None
            and getattr(current_settings, 'log_color', True)
            and hasattr(sys.stdout, 'isatty')
            and sys.stdout.isatty()
        )

        if use_colors:
            level_color = self.COLORS.get(record.levelname, '')
            formatted_json = json.dumps(log_dict, ensure_ascii=False)
            return f"{level_color}{formatted_json}{self.RESET}"
        else:
            return json.dumps(log_dict, ensure_ascii=False)

    def _get_log_dict(self, record: logging.LogRecord) -> dict:
        """Extract log dictionary from record, similar to JSONFormatter logic."""
        header = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            header["detail"] = dataclasses.asdict(log_payload)
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = {
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "stack_trace": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                }

        # Remove stack_trace for console output
        if (
            "detail" in header
            and "error" in header["detail"]
            and header["detail"]["error"]
        ):
            if "stack_trace" in header["detail"]["error"]:
                del header["detail"]["error"]["stack_trace"]
        elif "error" in header and header["error"]:
            if "stack_trace" in header["error"]:
                del header["error"]["stack_trace"]

        return header


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        header = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            header["detail"] = dataclasses.asdict(log_payload)
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = {
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "stack_trace": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                }
        return json.dumps(header, ensure_ascii=False)


class ConsoleJSONFormatter(JSONFormatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = json.loads(super().format(record))
        if (
            "detail" in log_dict
            and "error" in log_dict["detail"]
            and log_dict["detail"]["error"]
        ):
            if "stack_trace" in log_dict["detail"]["error"]:
                del log_dict["detail"]["error"]["stack_trace"]
        elif "error" in log_dict and log_dict["error"]:
            if "stack_trace" in log_dict["error"]:
                del log_dict["error"]["stack_trace"]
        return json.dumps(log_dict)


# Create the logging configuration dictionary
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {"()": JSONFormatter},
        "console_json": {"()": ConsoleJSONFormatter},
        "colored_console": {"()": ColoredConsoleFormatter},
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "colored_console",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "WARNING"},
        settings.app_name: {
            "handlers": ["default"],
            "level": settings.log_level.upper(),
            "propagate": False,
        },
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Add file handler if log_file_path is configured
if settings.log_file_path:
    try:
        # 如果是相对路径，相对于项目根目录（src的上级目录）
        if not os.path.isabs(settings.log_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_file_path = os.path.join(project_root, settings.log_file_path)
        else:
            log_file_path = settings.log_file_path

        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Add file handler to the configuration
        log_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "json",
            "filename": log_file_path,
            "mode": "a",
        }

        # Add file handler to the main logger
        log_config["loggers"][settings.app_name]["handlers"].append("file")

    except Exception as e:
        _error_console.print(
            f"Failed to configure file logging to {settings.log_file_path}: {e}"
        )

dictConfig(log_config)


class LogEvent(enum.Enum):
    MODEL_SELECTION = "model_selection"
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILURE = "request_failure"
    ANTHROPIC_REQUEST = "anthropic_body"
    OPENAI_REQUEST = "openai_request"
    OPENAI_RESPONSE = "openai_response"
    ANTHROPIC_RESPONSE = "anthropic_response"
    STREAMING_REQUEST = "streaming_request"
    STREAM_INTERRUPTED = "stream_interrupted"
    TOKEN_COUNT = "token_count"
    TOKEN_ENCODER_LOAD_FAILED = "token_encoder_load_failed"
    SYSTEM_PROMPT_ADJUSTED = "system_prompt_adjusted"
    TOOL_INPUT_SERIALIZATION_FAILURE = "tool_input_serialization_failure"
    IMAGE_FORMAT_UNSUPPORTED = "image_format_unsupported"
    MESSAGE_FORMAT_NORMALIZED = "message_format_normalized"
    TOOL_RESULT_SERIALIZATION_FAILURE = "tool_result_serialization_failure"
    TOOL_RESULT_PROCESSING = "tool_result_processing"
    TOOL_CHOICE_UNSUPPORTED = "tool_choice_unsupported"
    TOOL_ARGS_TYPE_MISMATCH = "tool_args_type_mismatch"
    TOOL_ARGS_PARSE_FAILURE = "tool_args_parse_failure"
    TOOL_ARGS_UNEXPECTED = "tool_args_unexpected"
    TOOL_ID_PLACEHOLDER = "tool_id_placeholder"
    TOOL_ID_UPDATED = "tool_id_updated"
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    HEALTH_CHECK = "health_check"
    PROVIDER_ERROR_DETAILS = "provider_error_details"
    REQUEST_RECEIVED = "request_received"





_logger = logging.getLogger(settings.app_name)


def _log(level: int, record: LogRecord, exc: Optional[Exception] = None) -> None:
    try:
        if exc:
            try:
                record.error = LogError(
                    name=type(exc).__name__,
                    message=str(exc),
                    stack_trace="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                    args=exc.args if hasattr(exc, "args") else tuple(),
                )
            except Exception:
                # If error processing fails, create minimal error info
                record.error = LogError(
                    name="ProcessingError",
                    message="Error occurred during exception processing",
                    stack_trace="",
                    args=tuple(),
                )

            if not record.message and str(exc):
                try:
                    record.message = str(exc)
                except Exception:
                    record.message = "An error occurred but message could not be extracted"
            elif not record.message:
                record.message = "An unspecified error occurred"

        _logger.log(level=level, msg=record.message, extra={"log_record": record})
    except Exception:
        # Last resort: use standard Python logging without custom formatting
        try:
            import logging as std_logging
            std_logging.getLogger("fallback").log(level, f"Log error: {record.message}")
        except Exception:
            pass  # Silent failure to prevent infinite recursion


def debug(record: LogRecord):
    _log(logging.DEBUG, record)


def info(record: LogRecord):
    _log(logging.INFO, record)


def warning(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.WARNING, record, exc=exc)


def error(record: LogRecord, exc: Optional[Exception] = None):
    try:
        # Note: Console traceback printing is disabled to keep console output clean
        # Full traceback details are still logged to file
        _log(logging.ERROR, record, exc=exc)
    except Exception:
        # Last resort: use standard Python logging
        try:
            import logging as std_logging
            std_logging.getLogger("fallback").error(f"Error logging failed: {record.message}")
        except Exception:
            pass  # Silent failure to prevent infinite recursion


def critical(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.CRITICAL, record, exc=exc)


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImageSource(BaseModel):
    type: str
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], List[Any]]
    is_error: Optional[bool] = None


ContentBlock = Union[
    ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult
]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(..., alias="input_schema")


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    @field_validator("top_k")
    def check_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            req_id = info.context.get("request_id") if info.context else None
            warning(
                LogRecord(
                    event=LogEvent.PARAMETER_UNSUPPORTED.value,
                    message="Parameter 'top_k' provided by client but is not directly supported by the OpenAI Chat Completions API and will be ignored.",
                    request_id=req_id,
                    data={"parameter": "top_k", "value": v},
                )
            )
        return v


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ProviderErrorMetadata(BaseModel):
    provider_name: str
    raw_error: Optional[Dict[str, Any]] = None


class AnthropicErrorType(str, enum.Enum):
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED = "overloaded_error"
    REQUEST_TOO_LARGE = "request_too_large_error"


class AnthropicErrorDetail(BaseModel):
    type: AnthropicErrorType
    message: str
    provider: Optional[str] = None
    provider_message: Optional[str] = None
    provider_code: Optional[Union[str, int]] = None


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


STATUS_CODE_ERROR_MAP: Dict[int, AnthropicErrorType] = {
    400: AnthropicErrorType.INVALID_REQUEST,
    401: AnthropicErrorType.AUTHENTICATION,
    403: AnthropicErrorType.PERMISSION,
    404: AnthropicErrorType.NOT_FOUND,
    413: AnthropicErrorType.REQUEST_TOO_LARGE,
    422: AnthropicErrorType.INVALID_REQUEST,
    429: AnthropicErrorType.RATE_LIMIT,
    500: AnthropicErrorType.API_ERROR,
    502: AnthropicErrorType.API_ERROR,
    503: AnthropicErrorType.OVERLOADED,
    504: AnthropicErrorType.API_ERROR,
}


def extract_provider_error_details(
    error_details_dict: Optional[Dict[str, Any]],
) -> Optional[ProviderErrorMetadata]:
    if not isinstance(error_details_dict, dict):
        return None
    metadata = error_details_dict.get("metadata")
    if not isinstance(metadata, dict):
        return None
    provider_name = metadata.get("provider_name")
    raw_error_str = metadata.get("raw")

    if not provider_name or not isinstance(provider_name, str):
        return None

    parsed_raw_error: Optional[Dict[str, Any]] = None
    if isinstance(raw_error_str, str):
        try:
            parsed_raw_error = json.loads(raw_error_str)
        except json.JSONDecodeError:
            warning(
                LogRecord(
                    event=LogEvent.PROVIDER_ERROR_DETAILS.value,
                    message=f"Failed to parse raw provider error string for {provider_name}.",
                )
            )
            parsed_raw_error = {"raw_string_parse_failed": raw_error_str}
    elif isinstance(raw_error_str, dict):
        parsed_raw_error = raw_error_str

    return ProviderErrorMetadata(
        provider_name=provider_name, raw_error=parsed_raw_error
    )


# Multi-provider client management

async def make_provider_request(provider: Provider, endpoint: str, data: Dict[str, Any], request_id: str, stream: bool = False, original_headers: Optional[Dict[str, str]] = None) -> Union[httpx.Response, Dict[str, Any]]:
    """Make a request to a specific provider"""
    url = provider_manager.get_request_url(provider, endpoint)
    headers = provider_manager.get_provider_headers(provider, original_headers)
    timeout = provider_manager.get_request_timeout()

    # Configure proxy if specified
    proxy_config = None
    if provider.proxy:
        proxy_config = provider.proxy

    debug(
        LogRecord(
            event="provider_request",
            message=f"Making request to provider: {provider.name}",
            request_id=request_id,
            data={"provider": provider.name, "url": url, "type": provider.type.value, "proxy": proxy_config, "headers": {k: v for k, v in headers.items() if k.lower() not in ['authorization', 'x-api-key']}}
        )
    )

    async with httpx.AsyncClient(timeout=timeout, proxy=proxy_config) as client:
        if stream:
            response = await client.post(url, json=data, headers=headers)
            # 在流式请求中，检查错误状态并记录详细信息
            if response.status_code >= 400:
                try:
                    error_text = response.text
                    if not error_text:
                        error_text = "Empty response body"
                    error_text = error_text[:1000]  # 限制长度
                except Exception:
                    error_text = "Failed to read response text"

                error(
                    LogRecord(
                        event="provider_http_error",
                        message=f"HTTP {response.status_code} error from provider: {provider.name}",
                        request_id=request_id,
                        data={
                            "provider": provider.name,
                            "status_code": response.status_code,
                            "content_type": response.headers.get("content-type", "unknown"),
                            "response_text": error_text,
                            "url": url,
                            "content_length": len(response.content)
                        }
                    )
                )
            response.raise_for_status()
            return response
        else:
            response = await client.post(url, json=data, headers=headers)

            # 记录响应状态和内容类型以便调试
            debug(
                LogRecord(
                    event="provider_response",
                    message=f"Received response from provider: {provider.name}",
                    request_id=request_id,
                    data={
                        "provider": provider.name,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", "unknown"),
                        "content_length": len(response.content)
                    }
                )
            )

            # 在非流式请求中，检查错误状态并记录详细信息
            if response.status_code >= 400:
                try:
                    error_text = response.text
                    if not error_text:
                        error_text = "Empty response body"
                    error_text = error_text[:1000]  # 限制长度
                except Exception:
                    error_text = "Failed to read response text"

                error(
                    LogRecord(
                        event="provider_http_error",
                        message=f"HTTP {response.status_code} error from provider: {provider.name}",
                        request_id=request_id,
                        data={
                            "provider": provider.name,
                            "status_code": response.status_code,
                            "content_type": response.headers.get("content-type", "unknown"),
                            "response_text": error_text,
                            "url": url,
                            "content_length": len(response.content)
                        }
                    )
                )

            response.raise_for_status()

            try:
                return response.json()
            except json.JSONDecodeError as e:
                # 记录响应内容以便调试
                response_text = response.text[:1000]  # 限制长度避免日志过长
                error(
                    LogRecord(
                        event="json_parse_error",
                        message=f"Failed to parse JSON response from provider: {provider.name}",
                        request_id=request_id,
                        data={
                            "provider": provider.name,
                            "status_code": response.status_code,
                            "content_type": response.headers.get("content-type", "unknown"),
                            "response_text": response_text,
                            "error": str(e)
                        }
                    ),
                    exc=e
                )
                raise e

async def make_anthropic_request(provider: Provider, messages_data: Dict[str, Any], request_id: str, stream: bool = False, original_headers: Optional[Dict[str, str]] = None) -> Union[httpx.Response, Dict[str, Any]]:
    """Make a request to an Anthropic-compatible provider"""
    return await make_provider_request(provider, "v1/messages", messages_data, request_id, stream, original_headers)

async def make_openai_request(provider: Provider, openai_params: Dict[str, Any], request_id: str, stream: bool = False, original_headers: Optional[Dict[str, str]] = None) -> Any:
    """Make a request to an OpenAI-compatible provider using openai client"""
    try:
        # 准备默认头部
        default_headers = {
            "HTTP-Referer": settings.referrer_url,
            "X-Title": settings.app_name,
        }

        # 如果提供了原始请求头，合并它们（排除需要替换的头部）
        if original_headers:
            for key, value in original_headers.items():
                # 跳过需要替换的认证相关头部、host头部和content-length头部
                if key.lower() not in ['authorization', 'x-api-key', 'host', 'content-length']:
                    default_headers[key] = value

        # Configure proxy if specified
        http_client = None
        if provider.proxy:
            http_client = httpx.AsyncClient(proxy=provider.proxy)

        # 处理auth_value的passthrough模式
        api_key_value = provider.auth_value
        if provider.auth_value == "passthrough" and original_headers:
            # 从原始请求头中提取认证token
            for key, value in original_headers.items():
                if key.lower() == "authorization":
                    # 提取Bearer token
                    if value.lower().startswith("bearer "):
                        api_key_value = value[7:]  # 移除"Bearer "前缀
                    else:
                        api_key_value = value
                    break
                elif key.lower() == "x-api-key":
                    api_key_value = value
                    break

            # 如果没有找到有效的认证头，使用一个占位符（openai客户端需要这个参数）
            if api_key_value == "passthrough":
                api_key_value = "placeholder-key"

        client = openai.AsyncClient(
            api_key=api_key_value,
            base_url=provider.base_url,
            default_headers=default_headers,
            timeout=provider_manager.get_request_timeout(),
            http_client=http_client,
        )

        if stream:
            return await client.chat.completions.create(**openai_params)
        else:
            return await client.chat.completions.create(**openai_params)
    except Exception as e:
        raise e

if not provider_manager:
    critical(
        LogRecord(
            event="provider_manager_init_failed",
            message="Failed to initialize provider manager",
        )
    )
    sys.exit(1)

# Add type check for all provider_manager calls
assert provider_manager is not None  # This will help with type checking


_token_encoder_cache: Dict[str, tiktoken.Encoding] = {}


def get_token_encoder(
    model_name: str = "gpt-4", request_id: Optional[str] = None
) -> tiktoken.Encoding:
    """Gets a tiktoken encoder, caching it for performance."""

    cache_key = "gpt-4"
    if cache_key not in _token_encoder_cache:
        try:
            _token_encoder_cache[cache_key] = tiktoken.encoding_for_model(cache_key)
        except Exception:
            try:
                _token_encoder_cache[cache_key] = tiktoken.get_encoding("cl100k_base")
                warning(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message=f"Could not load tiktoken encoder for '{cache_key}', using 'cl100k_base'. Token counts may be approximate.",
                        request_id=request_id,
                        data={"model_tried": cache_key},
                    )
                )
            except Exception as e_cl:
                critical(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message="Failed to load any tiktoken encoder (gpt-4, cl100k_base). Token counting will be inaccurate.",
                        request_id=request_id,
                    ),
                    exc=e_cl,
                )

                class DummyEncoder:
                    def encode(self, text: str) -> List[int]:
                        return list(range(len(text)))

                _token_encoder_cache[cache_key] = DummyEncoder()
    return _token_encoder_cache[cache_key]


def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
) -> int:
    enc = get_token_encoder(model_name, request_id)
    total_tokens = 0

    if isinstance(system, str):
        total_tokens += len(enc.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, SystemContent) and block.type == "text":
                total_tokens += len(enc.encode(block.text))

    for msg in messages:
        total_tokens += 4
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
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool input for token counting.",
                                data={"tool_name": block.name},
                                request_id=request_id,
                            )
                        )
                elif isinstance(block, ContentBlockToolResult):
                    try:
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            for item in block.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content_str += item.get("text", "")
                                else:
                                    content_str += json.dumps(item)
                        else:
                            content_str = json.dumps(block.content)
                        total_tokens += len(enc.encode(content_str))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool result for token counting.",
                                request_id=request_id,
                            )
                        )

    if tools:
        total_tokens += 2
        for tool in tools:
            total_tokens += len(enc.encode(tool.name))
            if tool.description:
                total_tokens += len(enc.encode(tool.description))
            try:
                schema_str = json.dumps(tool.input_schema)
                total_tokens += len(enc.encode(schema_str))
            except Exception:
                warning(
                    LogRecord(
                        event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                        message="Failed to serialize tool schema for token counting.",
                        data={"tool_name": tool.name},
                        request_id=request_id,
                    )
                )
    debug(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Estimated {total_tokens} input tokens for model {model_name}",
            data={"model": model_name, "token_count": total_tokens},
            request_id=request_id,
        )
    )
    return total_tokens


StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


def _serialize_tool_result_content_for_openai(
    anthropic_tool_result_content: Union[str, List[Dict[str, Any]], List[Any]],
    request_id: Optional[str],
    log_context: Dict,
) -> str:
    """
    Serializes Anthropic tool result content (which can be complex) into a single string
    as expected by OpenAI for the 'content' field of a 'tool' role message.
    """
    if isinstance(anthropic_tool_result_content, str):
        return anthropic_tool_result_content

    if isinstance(anthropic_tool_result_content, list):
        processed_parts = []
        contains_non_text_block = False
        for item in anthropic_tool_result_content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                processed_parts.append(str(item["text"]))
            else:
                try:
                    processed_parts.append(json.dumps(item))
                    contains_non_text_block = True
                except TypeError:
                    processed_parts.append(
                        f"<unserializable_item type='{type(item).__name__}'>"
                    )
                    contains_non_text_block = True

        result_str = "\n".join(processed_parts)
        if contains_non_text_block:
            warning(
                LogRecord(
                    event=LogEvent.TOOL_RESULT_PROCESSING.value,
                    message="Tool result content list contained non-text or complex items; parts were JSON stringified.",
                    request_id=request_id,
                    data={**log_context, "result_str_preview": result_str[:100]},
                )
            )
        return result_str

    try:
        return json.dumps(anthropic_tool_result_content)
    except TypeError as e:
        warning(
            LogRecord(
                event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                message=f"Failed to serialize tool result content to JSON: {e}. Returning error JSON.",
                request_id=request_id,
                data=log_context,
            )
        )
        return json.dumps(
            {
                "error": "Serialization failed",
                "original_type": str(type(anthropic_tool_result_content)),
            }
        )


def convert_anthropic_to_openai_messages(
    anthropic_messages: List[Message],
    anthropic_system: Optional[Union[str, List[SystemContent]]] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    system_text_content = ""
    if isinstance(anthropic_system, str):
        system_text_content = anthropic_system
    elif isinstance(anthropic_system, list):
        system_texts = [
            block.text
            for block in anthropic_system
            if isinstance(block, SystemContent) and block.type == "text"
        ]
        if len(system_texts) < len(anthropic_system):
            warning(
                LogRecord(
                    event=LogEvent.SYSTEM_PROMPT_ADJUSTED.value,
                    message="Non-text content blocks in Anthropic system prompt were ignored.",
                    request_id=request_id,
                )
            )
        system_text_content = "\n".join(system_texts)

    if system_text_content:
        openai_messages.append({"role": "system", "content": system_text_content})

    for i, msg in enumerate(anthropic_messages):
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            openai_parts_for_user_message = []
            assistant_tool_calls = []
            text_content_for_assistant = []

            if not content and role == "user":
                openai_messages.append({"role": "user", "content": ""})
                continue
            if not content and role == "assistant":
                openai_messages.append({"role": "assistant", "content": ""})
                continue

            for block_idx, block in enumerate(content):
                block_log_ctx = {
                    "anthropic_message_index": i,
                    "block_index": block_idx,
                    "block_type": block.type,
                }

                if isinstance(block, ContentBlockText):
                    if role == "user":
                        openai_parts_for_user_message.append(
                            {"type": "text", "text": block.text}
                        )
                    elif role == "assistant":
                        text_content_for_assistant.append(block.text)

                elif isinstance(block, ContentBlockImage) and role == "user":
                    if block.source.type == "base64":
                        openai_parts_for_user_message.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.source.media_type};base64,{block.source.data}"
                                },
                            }
                        )
                    else:
                        warning(
                            LogRecord(
                                event=LogEvent.IMAGE_FORMAT_UNSUPPORTED.value,
                                message=f"Image block with source type '{block.source.type}' (expected 'base64') ignored in user message {i}.",
                                request_id=request_id,
                                data=block_log_ctx,
                            )
                        )

                elif isinstance(block, ContentBlockToolUse) and role == "assistant":
                    try:
                        args_str = json.dumps(block.input)
                    except Exception as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message=f"Failed to serialize tool input for tool '{block.name}'. Using empty JSON.",
                                request_id=request_id,
                                data={
                                    **block_log_ctx,
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                },
                            ),
                            exc=e,
                        )
                        args_str = "{}"

                    assistant_tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": args_str},
                        }
                    )

                elif isinstance(block, ContentBlockToolResult) and role == "user":
                    serialized_content = _serialize_tool_result_content_for_openai(
                        block.content, request_id, block_log_ctx
                    )
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": serialized_content,
                        }
                    )

            if role == "user" and openai_parts_for_user_message:
                is_multimodal = any(
                    part["type"] == "image_url"
                    for part in openai_parts_for_user_message
                )
                if is_multimodal or len(openai_parts_for_user_message) > 1:
                    openai_messages.append(
                        {"role": "user", "content": openai_parts_for_user_message}
                    )
                elif (
                    len(openai_parts_for_user_message) == 1
                    and openai_parts_for_user_message[0]["type"] == "text"
                ):
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": openai_parts_for_user_message[0]["text"],
                        }
                    )
                elif not openai_parts_for_user_message:
                    openai_messages.append({"role": "user", "content": ""})

            if role == "assistant":
                assistant_text = "\n".join(filter(None, text_content_for_assistant))
                if assistant_text:
                    openai_messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )

                if assistant_tool_calls:
                    if (
                        openai_messages
                        and openai_messages[-1]["role"] == "assistant"
                        and openai_messages[-1].get("content")
                    ):
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

                    elif (
                        openai_messages
                        and openai_messages[-1]["role"] == "assistant"
                        and not openai_messages[-1].get("tool_calls")
                    ):
                        openai_messages[-1]["tool_calls"] = assistant_tool_calls
                        openai_messages[-1]["content"] = None
                    else:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

    final_openai_messages = []
    for msg_dict in openai_messages:
        if (
            msg_dict.get("role") == "assistant"
            and msg_dict.get("tool_calls")
            and msg_dict.get("content") is not None
        ):
            warning(
                LogRecord(
                    event=LogEvent.MESSAGE_FORMAT_NORMALIZED.value,
                    message="Corrected assistant message with tool_calls to have content: None.",
                    request_id=request_id,
                    data={"original_content": msg_dict["content"]},
                )
            )
            msg_dict["content"] = None
        final_openai_messages.append(msg_dict)

    return final_openai_messages


def convert_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[Tool]],
) -> Optional[List[Dict[str, Any]]]:
    if not anthropic_tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in anthropic_tools
    ]


def convert_anthropic_tool_choice_to_openai(
    anthropic_choice: Optional[ToolChoice],
    request_id: Optional[str] = None,
) -> Optional[Union[str, Dict[str, Any]]]:
    if not anthropic_choice:
        return None
    if anthropic_choice.type == "auto":
        return "auto"
    if anthropic_choice.type == "any":
        warning(
            LogRecord(
                event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
                message="Anthropic tool_choice type 'any' mapped to OpenAI 'auto'. Exact behavior might differ (OpenAI 'auto' allows no tool use).",
                request_id=request_id,
                data={"anthropic_tool_choice": anthropic_choice.model_dump()},
            )
        )
        return "auto"
    if anthropic_choice.type == "none":
        return "none"
    if anthropic_choice.type == "tool" and anthropic_choice.name:
        return {"type": "function", "function": {"name": anthropic_choice.name}}

    warning(
        LogRecord(
            event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
            message=f"Unsupported Anthropic tool_choice: {anthropic_choice.model_dump()}. Defaulting to 'auto'.",
            request_id=request_id,
            data={"anthropic_tool_choice": anthropic_choice.model_dump()},
        )
    )
    return "auto"


def convert_openai_to_anthropic_response(
    openai_response: openai.types.chat.ChatCompletion,
    original_anthropic_model_name: str,
    request_id: Optional[str] = None,
) -> MessagesResponse:
    anthropic_content: List[ContentBlock] = []
    anthropic_stop_reason: StopReasonType = None

    stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: "end_turn",
    }

    if openai_response.choices:
        choice = openai_response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        anthropic_stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        if message.content:
            anthropic_content.append(
                ContentBlockText(type="text", text=message.content)
            )

        if message.tool_calls:
            for call in message.tool_calls:
                if call.type == "function":
                    tool_input_dict: Dict[str, Any] = {}
                    try:
                        parsed_input = json.loads(call.function.arguments)
                        if isinstance(parsed_input, dict):
                            tool_input_dict = parsed_input
                        else:
                            tool_input_dict = {"value": parsed_input}
                            warning(
                                LogRecord(
                                    event=LogEvent.TOOL_ARGS_TYPE_MISMATCH.value,
                                    message=f"OpenAI tool arguments for '{call.function.name}' parsed to non-dict type '{type(parsed_input).__name__}'. Wrapped in 'value'.",
                                    request_id=request_id,
                                    data={
                                        "tool_name": call.function.name,
                                        "tool_id": call.id,
                                    },
                                )
                            )
                    except json.JSONDecodeError as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                                message=f"Failed to parse JSON arguments for tool '{call.function.name}'. Storing raw string.",
                                request_id=request_id,
                                data={
                                    "tool_name": call.function.name,
                                    "tool_id": call.id,
                                    "raw_args": call.function.arguments,
                                },
                            ),
                            exc=e,
                        )
                        tool_input_dict = {
                            "error_parsing_arguments": call.function.arguments
                        }

                    anthropic_content.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=call.id,
                            name=call.function.name,
                            input=tool_input_dict,
                        )
                    )
            if finish_reason == "tool_calls":
                anthropic_stop_reason = "tool_use"

    if not anthropic_content:
        anthropic_content.append(ContentBlockText(type="text", text=""))

    usage = openai_response.usage
    anthropic_usage = Usage(
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )

    response_id = (
        f"msg_{openai_response.id}"
        if openai_response.id
        else f"msg_{request_id}_completed"
    )

    return MessagesResponse(
        id=response_id,
        type="message",
        role="assistant",
        model=original_anthropic_model_name,
        content=anthropic_content,
        stop_reason=anthropic_stop_reason,
        usage=anthropic_usage,
    )


def _get_anthropic_error_details_from_exc(
    exc: Exception,
) -> Tuple[AnthropicErrorType, str, int, Optional[ProviderErrorMetadata]]:
    """Maps caught exceptions to Anthropic error type, message, status code, and provider details."""
    error_type = AnthropicErrorType.API_ERROR
    error_message = str(exc)
    status_code = 500
    provider_details: Optional[ProviderErrorMetadata] = None

    if isinstance(exc, openai.APIError):
        error_message = exc.message or str(exc)
        status_code = exc.status_code or 500
        error_type = STATUS_CODE_ERROR_MAP.get(
            status_code, AnthropicErrorType.API_ERROR
        )

        if hasattr(exc, "body") and isinstance(exc.body, dict):
            actual_error_details = exc.body.get("error", exc.body)
            provider_details = extract_provider_error_details(actual_error_details)

    if isinstance(exc, openai.AuthenticationError):
        error_type = AnthropicErrorType.AUTHENTICATION
    elif isinstance(exc, openai.RateLimitError):
        error_type = AnthropicErrorType.RATE_LIMIT
    elif isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        error_type = AnthropicErrorType.INVALID_REQUEST
    elif isinstance(exc, openai.PermissionDeniedError):
        error_type = AnthropicErrorType.PERMISSION
    elif isinstance(exc, openai.NotFoundError):
        error_type = AnthropicErrorType.NOT_FOUND

    return error_type, error_message, status_code, provider_details


def _format_anthropic_error_sse_event(
    error_type: AnthropicErrorType,
    message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> str:
    """Formats an error into the Anthropic SSE 'error' event structure."""
    anthropic_err_detail = AnthropicErrorDetail(type=error_type, message=message)
    if provider_details:
        anthropic_err_detail.provider = provider_details.provider_name
        if provider_details.raw_error and isinstance(
            provider_details.raw_error.get("error"), dict
        ):
            prov_err_obj = provider_details.raw_error["error"]
            anthropic_err_detail.provider_message = prov_err_obj.get("message")
            anthropic_err_detail.provider_code = prov_err_obj.get("code")
        elif provider_details.raw_error and isinstance(
            provider_details.raw_error.get("message"), str
        ):
            anthropic_err_detail.provider_message = provider_details.raw_error.get(
                "message"
            )
            anthropic_err_detail.provider_code = provider_details.raw_error.get("code")

    error_response = AnthropicErrorResponse(error=anthropic_err_detail)
    return f"event: error\ndata: {error_response.model_dump_json()}\n\n"


async def handle_anthropic_streaming_response_from_openai_stream(
    openai_stream: openai.AsyncStream[openai.types.chat.ChatCompletionChunk],
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
    success_callback: Optional[Callable[[], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Consumes an OpenAI stream and yields Anthropic-compatible SSE events.
    BUGFIX: Correctly handles content block indexing for mixed text/tool_use.
    """
    import asyncio

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"

    next_anthropic_block_idx = 0
    text_block_anthropic_idx: Optional[int] = None

    openai_tool_idx_to_anthropic_block_idx: Dict[int, int] = {}

    tool_states: Dict[int, Dict[str, Any]] = {}

    sent_tool_block_starts: set[int] = set()

    output_token_count = 0
    final_anthropic_stop_reason: StopReasonType = None

    enc = get_token_encoder(original_anthropic_model_name, request_id)

    openai_to_anthropic_stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = 200
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value
    
    # 添加超时控制
    streaming_start_time = time.time()
    last_activity = time.time()
    # 从配置获取超时设置，如果provider_manager不可用则使用默认值
    streaming_timeout = provider_manager.settings.get('streaming_timeout', 120) if provider_manager else 120
    streaming_activity_timeout = provider_manager.settings.get('streaming_activity_timeout', 60) if provider_manager else 60
    
    info(
        LogRecord(
            "openai_streaming_start",
            f"Starting OpenAI stream conversion with timeout protection",
            request_id,
            {
                "anthropic_message_id": anthropic_message_id,
                "streaming_timeout": streaming_timeout,
                "streaming_activity_timeout": streaming_activity_timeout
            },
        )
    )

    try:
        message_start_event_data = {
            "type": "message_start",
            "message": {
                "id": anthropic_message_id,
                "type": "message",
                "role": "assistant",
                "model": original_anthropic_model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_event_data)}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for chunk in openai_stream:
            # 检查超时
            current_time = time.time()
            if current_time - streaming_start_time > streaming_timeout:
                info(
                    LogRecord(
                        "openai_streaming_timeout",
                        f"OpenAI stream timeout",
                        request_id,
                        {
                            "timeout_duration": streaming_timeout,
                            "elapsed_time": current_time - streaming_start_time,
                            "phase": "openai_streaming"
                        },
                    )
                )
                raise asyncio.TimeoutError(f"OpenAI streaming timeout after {streaming_timeout}s")
            
            # 检查活动超时
            if current_time - last_activity > streaming_activity_timeout:
                info(
                    LogRecord(
                        "openai_streaming_activity_timeout",
                        f"OpenAI stream activity timeout",
                        request_id,
                        {
                            "activity_timeout": streaming_activity_timeout,
                            "inactive_duration": current_time - last_activity,
                            "elapsed_time": current_time - streaming_start_time
                        },
                    )
                )
                raise asyncio.TimeoutError(f"OpenAI streaming activity timeout after {streaming_activity_timeout}s of inactivity")
            
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            openai_finish_reason = chunk.choices[0].finish_reason
            
            # 更新活动时间
            last_activity = current_time

            if delta.content:
                output_token_count += len(enc.encode(delta.content))
                if text_block_anthropic_idx is None:
                    text_block_anthropic_idx = next_anthropic_block_idx
                    next_anthropic_block_idx += 1
                    start_text_event = {
                        "type": "content_block_start",
                        "index": text_block_anthropic_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(start_text_event)}\n\n"

                text_delta_event = {
                    "type": "content_block_delta",
                    "index": text_block_anthropic_idx,
                    "delta": {"type": "text_delta", "text": delta.content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(text_delta_event)}\n\n"

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    openai_tc_idx = tool_delta.index

                    if openai_tc_idx not in openai_tool_idx_to_anthropic_block_idx:
                        current_anthropic_tool_block_idx = next_anthropic_block_idx
                        next_anthropic_block_idx += 1
                        openai_tool_idx_to_anthropic_block_idx[openai_tc_idx] = (
                            current_anthropic_tool_block_idx
                        )

                        tool_states[current_anthropic_tool_block_idx] = {
                            "id": tool_delta.id
                            or f"tool_ph_{request_id}_{current_anthropic_tool_block_idx}",
                            "name": "",
                            "arguments_buffer": "",
                        }
                        if not tool_delta.id:
                            warning(
                                LogRecord(
                                    LogEvent.TOOL_ID_PLACEHOLDER.value,
                                    f"Generated placeholder Tool ID for OpenAI tool index {openai_tc_idx} -> Anthropic block {current_anthropic_tool_block_idx}",
                                    request_id,
                                )
                            )
                    else:
                        current_anthropic_tool_block_idx = (
                            openai_tool_idx_to_anthropic_block_idx[openai_tc_idx]
                        )

                    tool_state = tool_states[current_anthropic_tool_block_idx]

                    if tool_delta.id and tool_state["id"].startswith("tool_ph_"):
                        debug(
                            LogRecord(
                                LogEvent.TOOL_ID_UPDATED.value,
                                f"Updated placeholder Tool ID for Anthropic block {current_anthropic_tool_block_idx} to {tool_delta.id}",
                                request_id,
                            )
                        )
                        tool_state["id"] = tool_delta.id

                    if tool_delta.function:
                        if tool_delta.function.name:
                            tool_state["name"] = tool_delta.function.name
                        if tool_delta.function.arguments:
                            tool_state["arguments_buffer"] += (
                                tool_delta.function.arguments
                            )
                            output_token_count += len(
                                enc.encode(tool_delta.function.arguments)
                            )

                    if (
                        current_anthropic_tool_block_idx not in sent_tool_block_starts
                        and tool_state["id"]
                        and not tool_state["id"].startswith("tool_ph_")
                        and tool_state["name"]
                    ):
                        start_tool_event = {
                            "type": "content_block_start",
                            "index": current_anthropic_tool_block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_state["id"],
                                "name": tool_state["name"],
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(start_tool_event)}\n\n"
                        sent_tool_block_starts.add(current_anthropic_tool_block_idx)

                    if (
                        tool_delta.function
                        and tool_delta.function.arguments
                        and current_anthropic_tool_block_idx in sent_tool_block_starts
                    ):
                        args_delta_event = {
                            "type": "content_block_delta",
                            "index": current_anthropic_tool_block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_delta.function.arguments,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(args_delta_event)}\n\n"

            if openai_finish_reason:
                final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(
                    openai_finish_reason, "end_turn"
                )
                if openai_finish_reason == "tool_calls":
                    final_anthropic_stop_reason = "tool_use"
                break

        if text_block_anthropic_idx is not None:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_anthropic_idx})}\n\n"

        for anthropic_tool_idx in sent_tool_block_starts:
            tool_state_to_finalize = tool_states.get(anthropic_tool_idx)
            if tool_state_to_finalize:
                try:
                    json.loads(tool_state_to_finalize["arguments_buffer"])
                except json.JSONDecodeError:
                    warning(
                        LogRecord(
                            event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                            message=f"Buffered arguments for tool '{tool_state_to_finalize.get('name')}' (Anthropic block {anthropic_tool_idx}) did not form valid JSON.",
                            request_id=request_id,
                            data={
                                "buffered_args": tool_state_to_finalize[
                                    "arguments_buffer"
                                ][:100]
                            },
                        )
                    )
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_tool_idx})}\n\n"

        if final_anthropic_stop_reason is None:
            final_anthropic_stop_reason = "end_turn"

        message_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_anthropic_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        # 流式传输成功完成，调用成功回调
        total_time = time.time() - streaming_start_time
        info(
            LogRecord(
                "openai_streaming_completed",
                f"OpenAI stream conversion completed successfully",
                request_id,
                {
                    "output_tokens": output_token_count,
                    "stop_reason": final_anthropic_stop_reason,
                    "streaming_duration": total_time,
                    "anthropic_message_id": anthropic_message_id
                },
            )
        )
        
        if success_callback:
            success_callback()

    except Exception as e:
        stream_status_code = 500
        stream_log_event = LogEvent.REQUEST_FAILURE.value
        error_type, error_msg_str, _, provider_err_details = (
            _get_anthropic_error_details_from_exc(e)
        )
        stream_final_message = f"Error during OpenAI stream conversion: {error_msg_str}"
        final_anthropic_stop_reason = "error"

        error(
            LogRecord(
                event=LogEvent.STREAM_INTERRUPTED.value,
                message=stream_final_message,
                request_id=request_id,
                data={
                    "error_type": error_type.value,
                    "provider_details": provider_err_details.model_dump()
                    if provider_err_details
                    else None,
                },
            ),
            exc=e,
        )
        yield _format_anthropic_error_sse_event(
            error_type, error_msg_str, provider_err_details
        )

    finally:
        duration_ms = (time.monotonic() - start_time_mono) * 1000
        log_data = {
            "status_code": stream_status_code,
            "duration_ms": duration_ms,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
            "stop_reason": final_anthropic_stop_reason,
        }
        if stream_log_event == LogEvent.REQUEST_COMPLETED.value:
            info(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
        else:
            error(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )


app = fastapi.FastAPI(
    title=settings.app_name,
    description="Routes Anthropic API requests to an OpenAI-compatible API, selecting models dynamically.",
    version=settings.app_version,
    docs_url=None,
    redoc_url=None,
)


def select_target_model_and_provider_options(client_model_name: str, request_id: str) -> List[Tuple[str, Provider]]:
    """Selects multiple target model and provider options based on the client's request."""
    if not provider_manager:
        raise RuntimeError("Provider manager not initialized")

    # 使用新的灵活选择方法
    options = provider_manager.select_model_and_provider_options(client_model_name)

    if not options:
        return []

    # 已经是 (target_model, provider) 格式
    result = options

    # 记录选择的选项
    debug(
        LogRecord(
            event=LogEvent.MODEL_SELECTION.value,
            message=f"Client model '{client_model_name}' has {len(result)} available options",
            request_id=request_id,
            data={
                "client_model": client_model_name,
                "available_options": [
                    {
                        "target_model": target_model,
                        "provider": provider.name,
                        "provider_type": provider.type.value
                    }
                    for target_model, provider in result
                ]
            },
        )
    )

    return result


def select_target_model_and_provider(client_model_name: str, request_id: str) -> Optional[Tuple[str, Provider]]:
    """Selects the target model and provider based on the client's request (backward compatibility)."""
    options = select_target_model_and_provider_options(client_model_name, request_id)
    if options:
        return options[0]  # 返回第一个（最高优先级的）选项
    return None


def _build_anthropic_error_response(
    error_type: AnthropicErrorType,
    message: str,
    status_code: int,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> JSONResponse:
    """Creates a JSONResponse with Anthropic-formatted error."""
    try:
        err_detail = AnthropicErrorDetail(type=error_type, message=message)
        if provider_details:
            try:
                err_detail.provider = provider_details.provider_name
                if provider_details.raw_error:
                    if isinstance(provider_details.raw_error, dict):
                        prov_err_obj = provider_details.raw_error.get("error")
                        if isinstance(prov_err_obj, dict):
                            err_detail.provider_message = prov_err_obj.get("message")
                            err_detail.provider_code = prov_err_obj.get("code")
                        elif isinstance(provider_details.raw_error.get("message"), str):
                            err_detail.provider_message = provider_details.raw_error.get(
                                "message"
                            )
                            err_detail.provider_code = provider_details.raw_error.get("code")
            except Exception:
                # If provider details processing fails, continue with basic error
                pass

        error_resp_model = AnthropicErrorResponse(error=err_detail)
        return JSONResponse(
            status_code=status_code, content=error_resp_model.model_dump(exclude_unset=True)
        )
    except Exception:
        # Emergency fallback - return minimal JSON response
        return JSONResponse(
            status_code=status_code,
            content={
                "type": "error",
                "error": {
                    "type": error_type.value if hasattr(error_type, 'value') else "api_error",
                    "message": message if isinstance(message, str) else "Internal server error"
                }
            }
        )


async def _log_and_return_error_response(
    request: Request,
    status_code: int,
    anthropic_error_type: AnthropicErrorType,
    error_message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
    caught_exception: Optional[Exception] = None,
) -> JSONResponse:
    try:
        request_id = getattr(request.state, "request_id", "unknown")
        start_time_mono = getattr(request.state, "start_time_monotonic", time.monotonic())
        duration_ms = (time.monotonic() - start_time_mono) * 1000

        log_data = {
            "status_code": status_code,
            "duration_ms": duration_ms,
            "error_type": anthropic_error_type.value,
            "client_ip": request.client.host if request.client else "unknown",
        }
        if provider_details:
            log_data["provider_name"] = provider_details.provider_name
            log_data["provider_raw_error"] = provider_details.raw_error

        # Protected error logging
        try:
            error(
                LogRecord(
                    event=LogEvent.REQUEST_FAILURE.value,
                    message=f"Request failed: {error_message}",
                    request_id=request_id,
                    data=log_data,
                ),
                exc=caught_exception,
            )
        except Exception:
            # If logging fails, continue with response generation
            pass

        return _build_anthropic_error_response(
            anthropic_error_type, error_message, status_code, provider_details
        )
    except Exception:
        # Emergency fallback - return minimal response
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Internal server error"
                }
            },
            status_code=500
        )


@app.post("/v1/messages", response_model=None, tags=["API"], status_code=200)
async def create_message_proxy(
    request: Request,
) -> Union[JSONResponse, StreamingResponse]:
    """
    Main endpoint for Anthropic message completions, proxied to an OpenAI-compatible API.
    Handles request/response conversions, streaming, and dynamic model selection.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time_monotonic = time.monotonic()
    request_signature = None  # 用于最终清理

    async def _complete_request_and_cleanup(result):
        """完成请求并清理去重状态"""
        _complete_and_cleanup_request(request_signature, result)
        return result

    # 打印所有的 request头信息
    debug(
        LogRecord(
            LogEvent.REQUEST_RECEIVED.value,
            "Received new request",
            request_id,
            {
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )
    )

    # 提取原始请求头
    original_headers = dict(request.headers)

    # 标记请求开始，用于智能provider恢复
    if provider_manager:
        provider_manager.mark_request_start()

    try:
        raw_body = await request.json()

        # 生成请求签名并检查是否为重复请求
        request_signature = _generate_request_signature(raw_body)
        duplicate_result = await _handle_duplicate_request(request_signature, request_id)

        if duplicate_result is not None:
            # 这是重复请求，返回原请求的结果
            return duplicate_result
        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Received Anthropic request body",
                request_id,
                {"body_summary": _create_body_summary(raw_body)},
            )
        )

        anthropic_request = MessagesRequest.model_validate(
            raw_body, context={"request_id": request_id}
        )
    except json.JSONDecodeError as e:
        response = await _log_and_return_error_response(
            request,
            400,
            AnthropicErrorType.INVALID_REQUEST,
            "Invalid JSON body.",
            caught_exception=e,
        )
        _complete_and_cleanup_request(request_signature, response)
        return response
    except ValidationError as e:
        response = await _log_and_return_error_response(
            request,
            422,
            AnthropicErrorType.INVALID_REQUEST,
            f"Invalid request body: {e.errors()}",
            caught_exception=e,
        )
        _complete_and_cleanup_request(request_signature, response)
        return response

    is_stream = anthropic_request.stream or False
    provider_options = select_target_model_and_provider_options(anthropic_request.model, request_id)

    if not provider_options:
        # No healthy providers available
        response = await _log_and_return_error_response(
            request,
            503,
            AnthropicErrorType.API_ERROR,
            "No providers available for the requested model. Please try again later.",
            caught_exception=None,
        )
        _complete_and_cleanup_request(request_signature, response)
        return response

    # 使用第一个选项开始处理，后面可以fallback到其他选项
    target_model_name, current_provider = provider_options[0]

    estimated_input_tokens = count_tokens_for_anthropic_request(
        messages=anthropic_request.messages,
        system=anthropic_request.system,
        model_name=anthropic_request.model,
        tools=anthropic_request.tools,
        request_id=request_id,
    )

    info(
        LogRecord(
            event=LogEvent.REQUEST_START.value,
            message="Processing new message request",
            request_id=request_id,
            data={
                "client_model": anthropic_request.model,
                "available_options": len(provider_options),
                "primary_target_model": target_model_name,
                "primary_provider": current_provider.name,
                "provider_type": current_provider.type.value,
                "stream": is_stream,
                "estimated_input_tokens": estimated_input_tokens,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )
    )

    # Multi-provider request handling with retry logic using available options
    if not provider_manager:
        response = await _log_and_return_error_response(
            request,
            503,
            AnthropicErrorType.API_ERROR,
            "Provider manager not available.",
            caught_exception=None,
        )
        _complete_and_cleanup_request(request_signature, response)
        return response

    # 使用available options进行重试，而不是传统的provider切换
    max_retries = len(provider_options)

    for attempt in range(max_retries):
        # 使用当前尝试的选项
        target_model_name, current_provider = provider_options[attempt]

        try:
            if current_provider.type == ProviderType.ANTHROPIC:
                # Direct Anthropic API request
                anthropic_data = anthropic_request.model_dump(exclude_unset=True)
                anthropic_data["model"] = target_model_name

                if is_stream:
                    debug(
                        LogRecord(
                            LogEvent.STREAMING_REQUEST.value,
                            f"Initiating streaming request to Anthropic provider: {current_provider.name}",
                            request_id,
                            {
                                "provider": current_provider.name,
                                "attempt": attempt + 1,
                                "is_fallback": attempt > 0,
                                "total_attempts": max_retries,
                            }
                        )
                    )
                    anthropic_response = await make_anthropic_request(
                        current_provider, anthropic_data, request_id, stream=True, original_headers=original_headers
                    )

                    # 验证流式响应是否有效
                    if not hasattr(anthropic_response, 'aiter_lines'):
                        raise Exception(f"Invalid streaming response from provider {current_provider.name}")

                    # 在开始流式传输之前预检查错误事件
                    # 读取前几行来检测是否有 error event
                    first_lines = []
                    line_iterator = anthropic_response.aiter_lines()

                    # 预读前几行检查错误（添加超时保护）
                    import asyncio
                    
                    preread_start_time = time.time()
                    preread_timeout = provider_manager.settings.get('streaming_timeout', 120) if provider_manager else 120
                    
                    info(
                        LogRecord(
                            "streaming_preread_start",
                            f"Starting pre-read phase for provider: {current_provider.name}",
                            request_id,
                            {
                                "provider": current_provider.name,
                                "preread_timeout": preread_timeout
                            },
                        )
                    )
                    
                    for i in range(5):  # 检查前5行
                        try:
                            # 添加超时检查
                            if time.time() - preread_start_time > preread_timeout:
                                info(
                                    LogRecord(
                                        "streaming_preread_timeout",
                                        f"Pre-read timeout for provider: {current_provider.name}",
                                        request_id,
                                        {
                                            "provider": current_provider.name,
                                            "timeout_duration": preread_timeout,
                                            "elapsed_time": time.time() - preread_start_time,
                                            "preread_lines": i
                                        },
                                    )
                                )
                                raise asyncio.TimeoutError(f"Pre-read timeout after {preread_timeout}s")
                            
                            line = await line_iterator.__anext__()
                            first_lines.append(line)
                            if line.strip() == "event: error":
                                # 读取下一行获取错误数据
                                try:
                                    data_line = await line_iterator.__anext__()
                                    first_lines.append(data_line)
                                    if data_line.startswith("data: "):
                                        try:
                                            error_data = json.loads(data_line[6:])  # 去掉 "data: " 前缀
                                            error_type = error_data.get("error", {}).get("type", "unknown_error")
                                            # 创建包含错误类型的异常
                                            error_msg = f"Provider {current_provider.name} returned error event in streaming response: {error_type}"
                                            streaming_error = Exception(error_msg)
                                            streaming_error.streaming_error_type = error_type
                                            raise streaming_error
                                        except json.JSONDecodeError:
                                            pass
                                except StopAsyncIteration:
                                    pass
                                raise Exception(f"Provider {current_provider.name} returned error event in streaming response")
                        except StopAsyncIteration:
                            break


                    # Handle Anthropic streaming response directly
                    async def anthropic_stream_generator():
                        import asyncio
                        
                        try:
                            line_count = 0
                            byte_count = 0
                            start_time = time.time()
                            last_activity = time.time()
                            # 从配置文件获取超时设置
                            streaming_timeout = provider_manager.settings.get('streaming_timeout', 120) if provider_manager else 120
                            streaming_activity_timeout = provider_manager.settings.get('streaming_activity_timeout', 30) if provider_manager else 30
                            
                            info(
                                LogRecord(
                                    "streaming_start",
                                    f"Starting stream generation for provider: {current_provider.name}",
                                    request_id,
                                    {
                                        "provider": current_provider.name, 
                                        "is_fallback": attempt > 0,
                                        "streaming_timeout": streaming_timeout,
                                        "streaming_activity_timeout": streaming_activity_timeout
                                    },
                                )
                            )

                            first_few_lines = []
                            last_few_lines = []
                            raw_lines = []  # 记录原始行数据用于调试


                            # 首先输出预读的行
                            for line in first_lines:
                                # 检查超时
                                if time.time() - start_time > streaming_timeout:
                                    info(
                                        LogRecord(
                                            "streaming_timeout",
                                            f"Stream timeout during pre-read lines for provider: {current_provider.name}",
                                            request_id,
                                            {
                                                "provider": current_provider.name,
                                                "timeout_duration": streaming_timeout,
                                                "elapsed_time": time.time() - start_time,
                                                "phase": "pre_read"
                                            },
                                        )
                                    )
                                    raise asyncio.TimeoutError(f"Streaming timeout after {streaming_timeout}s during pre-read")
                                
                                line_count += 1
                                byte_count += len(line)
                                last_activity = time.time()

                                # 保存前几行和后几行用于调试
                                if len(first_few_lines) < 5:
                                    first_few_lines.append(line)
                                raw_lines.append(line)

                                # 输出预读的行
                                yield f"{line}\n"

                            # 然后处理剩余的响应
                            async for line in line_iterator:
                                # 检查超时
                                current_time = time.time()
                                if current_time - start_time > streaming_timeout:
                                    info(
                                        LogRecord(
                                            "streaming_timeout",
                                            f"Stream timeout during streaming for provider: {current_provider.name}",
                                            request_id,
                                            {
                                                "provider": current_provider.name,
                                                "timeout_duration": streaming_timeout,
                                                "elapsed_time": current_time - start_time,
                                                "phase": "streaming"
                                            },
                                        )
                                    )
                                    raise asyncio.TimeoutError(f"Streaming timeout after {streaming_timeout}s")
                                
                                # 检查活动超时（无新数据）
                                if current_time - last_activity > streaming_activity_timeout:
                                    info(
                                        LogRecord(
                                            "streaming_activity_timeout",
                                            f"Stream activity timeout for provider: {current_provider.name}",
                                            request_id,
                                            {
                                                "provider": current_provider.name,
                                                "activity_timeout": streaming_activity_timeout,
                                                "inactive_duration": current_time - last_activity,
                                                "elapsed_time": current_time - start_time
                                            },
                                        )
                                    )
                                    raise asyncio.TimeoutError(f"Streaming activity timeout after {streaming_activity_timeout}s of inactivity")
                                # 不要过滤空行！SSE 格式需要空行作为事件分隔符
                                line_count += 1
                                byte_count += len(line)
                                last_activity = current_time

                                # 错误检测已经在预检查阶段完成，这里不需要重复检查

                                # 记录原始行数据（前10行）用于调试
                                if len(raw_lines) < 10:
                                    raw_lines.append(repr(line)[:300])  # 使用repr显示原始格式

                                # 记录前几行用于调试（只记录非空行）
                                if line.strip() and len(first_few_lines) < 3:
                                    first_few_lines.append(line.strip()[:200])

                                # 记录最后几行用于调试流结束格式
                                if line.strip():
                                    last_few_lines.append(line.strip()[:200])
                                    if len(last_few_lines) > 5:
                                        last_few_lines.pop(0)

                                yield f"{line}\n"


                            # 错误检测已经在预检查阶段完成，能到这里说明流式传输成功

                            # 记录原始行数据用于对比不同provider的响应格式
                            if raw_lines:
                                debug(
                                    LogRecord(
                                        event=LogEvent.STREAMING_REQUEST.value,
                                        message=f"Raw streaming lines from {current_provider.name}",
                                        request_id=request_id,
                                        data={"provider": current_provider.name, "raw_lines": raw_lines}
                                    )
                                )

                            # 记录流式数据的开头和结尾部分用于调试
                            if first_few_lines:
                                debug(
                                    LogRecord(
                                        "streaming_sample",
                                        f"First few lines from {current_provider.name}",
                                        request_id,
                                        {"provider": current_provider.name, "first_lines": first_few_lines},
                                    )
                                )

                            if last_few_lines:
                                debug(
                                    LogRecord(
                                        "streaming_end_sample",
                                        f"Last few lines from {current_provider.name}",
                                        request_id,
                                        {"provider": current_provider.name, "last_lines": last_few_lines},
                                    )
                                )

                            # 只有在流式传输完全成功后才标记成功
                            end_time = time.time()
                            total_time = end_time - start_time
                            
                            current_provider.mark_success()
                            if provider_manager:
                                provider_manager.mark_provider_success(current_provider.name)
                            info(
                                LogRecord(
                                    LogEvent.REQUEST_COMPLETED.value,
                                    f"Streaming request completed successfully via provider: {current_provider.name}",
                                    request_id,
                                    data={
                                        "status_code": 200,
                                        "provider": current_provider.name,
                                        "lines_streamed": line_count,
                                        "bytes_streamed": byte_count,
                                        "is_fallback": attempt > 0,
                                        "attempt_number": attempt + 1,
                                        "fallback_reason": "provider_failure" if attempt > 0 else "primary_success",
                                        "streaming_duration": total_time
                                    },
                                )
                            )
                        except Exception as e:
                            # 流式传输过程中出现异常
                            elapsed_time = time.time() - start_time
                            
                            error(
                                LogRecord(
                                    "streaming_error",
                                    f"Streaming failed for provider: {current_provider.name}",
                                    request_id,
                                    data={
                                        "provider": current_provider.name, 
                                        "error": str(e), 
                                        "elapsed_time": elapsed_time,
                                        "error_type": type(e).__name__
                                    },
                                ),
                                exc=e
                            )
                            # 注意：这里不能标记 provider 失败，因为响应已经开始返回给客户端
                            # 只能记录错误，让客户端处理重试
                            # 发送符合Anthropic格式的错误事件
                            error_event = {
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": f"Streaming interrupted: {str(e)}"
                                }
                            }
                            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

                    # 对于流式响应，需要完成去重状态处理
                    async def stream_with_cleanup():
                        """流式生成器包装器，完成后自动清理去重状态"""
                        try:
                            async for chunk in anthropic_stream_generator():
                                yield chunk
                            # 流式传输成功完成
                            _complete_and_cleanup_request(request_signature, "streaming_success")
                        except Exception as e:
                            # 流式传输失败，设置失败状态
                            _complete_and_cleanup_request(request_signature, Exception(f"Streaming failed: {str(e)}"))
                            raise

                    streaming_response = StreamingResponse(
                        stream_with_cleanup(),
                        media_type="text/event-stream",
                    )
                    return streaming_response
                else:
                    debug(
                        LogRecord(
                            LogEvent.ANTHROPIC_REQUEST.value,
                            f"Sending non-streaming request to Anthropic provider: {current_provider.name}",
                            request_id,
                        )
                    )
                    anthropic_response_data = await make_anthropic_request(
                        current_provider, anthropic_data, request_id, stream=False, original_headers=original_headers
                    )

                    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
                    info(
                        LogRecord(
                            event=LogEvent.REQUEST_COMPLETED.value,
                            message=f"Non-streaming request completed successfully via provider: {current_provider.name}",
                            request_id=request_id,
                            data={
                                "status_code": 200,
                                "duration_ms": duration_ms,
                                "provider": current_provider.name,
                            },
                        )
                    )

                    current_provider.mark_success()
                    if provider_manager:
                        provider_manager.mark_provider_success(current_provider.name)
                    
                    # 非流式请求完成
                    response = JSONResponse(content=anthropic_response_data)
                    return await _complete_request_and_cleanup(response)

            else:  # OpenAI-compatible provider
                try:
                    openai_messages = convert_anthropic_to_openai_messages(
                        anthropic_request.messages, anthropic_request.system, request_id=request_id
                    )
                    openai_tools = convert_anthropic_tools_to_openai(anthropic_request.tools)
                    openai_tool_choice = convert_anthropic_tool_choice_to_openai(
                        anthropic_request.tool_choice, request_id
                    )
                except Exception as e:
                    return await _log_and_return_error_response(
                        request,
                        500,
                        AnthropicErrorType.API_ERROR,
                        "Error during request conversion.",
                        caught_exception=e,
                    )

                openai_params: Dict[str, Any] = {
                    "model": target_model_name,
                    "messages": cast(List[ChatCompletionMessageParam], openai_messages),
                    "max_tokens": anthropic_request.max_tokens,
                    "stream": is_stream,
                }
                if anthropic_request.temperature is not None:
                    openai_params["temperature"] = anthropic_request.temperature
                if anthropic_request.top_p is not None:
                    openai_params["top_p"] = anthropic_request.top_p
                if anthropic_request.stop_sequences:
                    openai_params["stop"] = anthropic_request.stop_sequences
                if openai_tools:
                    openai_params["tools"] = cast(
                        Optional[List[ChatCompletionToolParam]], openai_tools
                    )
                if openai_tool_choice:
                    openai_params["tool_choice"] = openai_tool_choice
                if anthropic_request.metadata and anthropic_request.metadata.get("user_id"):
                    user_id = str(anthropic_request.metadata.get("user_id"))
                    # OpenRouter has a 128 character limit on the user field
                    if len(user_id) > 128:
                        user_id = user_id[:128]
                    openai_params["user"] = user_id

                debug(
                    LogRecord(
                        LogEvent.OPENAI_REQUEST.value,
                        f"Prepared OpenAI request parameters for provider: {current_provider.name}",
                        request_id,
                        {"params": openai_params},
                    )
                )

                if is_stream:
                    debug(
                        LogRecord(
                            LogEvent.STREAMING_REQUEST.value,
                            f"Initiating streaming request to OpenAI-compatible provider: {current_provider.name}",
                            request_id,
                        )
                    )
                    openai_stream_response = await make_openai_request(
                        current_provider, openai_params, request_id, stream=True, original_headers=original_headers
                    )

                    # 定义成功回调函数
                    def on_stream_success():
                        current_provider.mark_success()
                        if provider_manager:
                            provider_manager.mark_provider_success(current_provider.name)

                    # 注意：不在这里 mark_success，而是在流式传输完成后
                    # 对于流式响应，需要完成去重状态处理
                    async def openai_stream_with_cleanup():
                        """OpenAI流式生成器包装器，完成后自动清理去重状态"""
                        try:
                            async for chunk in handle_anthropic_streaming_response_from_openai_stream(
                                openai_stream_response,
                                anthropic_request.model,
                                estimated_input_tokens,
                                request_id,
                                request.state.start_time_monotonic,
                                success_callback=on_stream_success,
                            ):
                                yield chunk
                            # 流式传输成功完成
                            _complete_and_cleanup_request(request_signature, "streaming_success")
                        except Exception as e:
                            # 流式传输失败，设置失败状态
                            _complete_and_cleanup_request(request_signature, Exception(f"Streaming failed: {str(e)}"))
                            raise

                    streaming_response = StreamingResponse(
                        openai_stream_with_cleanup(),
                        media_type="text/event-stream",
                    )
                    return streaming_response
                else:
                    debug(
                        LogRecord(
                            LogEvent.OPENAI_REQUEST.value,
                            f"Sending non-streaming request to OpenAI-compatible provider: {current_provider.name}",
                            request_id,
                        )
                    )
                    openai_response_obj = await make_openai_request(
                        current_provider, openai_params, request_id, stream=False, original_headers=original_headers
                    )

                    debug(
                        LogRecord(
                            LogEvent.OPENAI_RESPONSE.value,
                            f"Received OpenAI response from provider: {current_provider.name}",
                            request_id,
                            {"response": openai_response_obj.model_dump()},
                        )
                    )

                    anthropic_response_obj = convert_openai_to_anthropic_response(
                        openai_response_obj, anthropic_request.model, request_id=request_id
                    )
                    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
                    info(
                        LogRecord(
                            event=LogEvent.REQUEST_COMPLETED.value,
                            message=f"Non-streaming request completed successfully via provider: {current_provider.name}",
                            request_id=request_id,
                            data={
                                "status_code": 200,
                                "duration_ms": duration_ms,
                                "input_tokens": anthropic_response_obj.usage.input_tokens,
                                "output_tokens": anthropic_response_obj.usage.output_tokens,
                                "stop_reason": anthropic_response_obj.stop_reason,
                                "provider": current_provider.name,
                            },
                        )
                    )
                    debug(
                        LogRecord(
                            LogEvent.ANTHROPIC_RESPONSE.value,
                            f"Prepared Anthropic response from provider: {current_provider.name}",
                            request_id,
                            {"response": anthropic_response_obj.model_dump(exclude_unset=True)},
                        )
                    )

                    current_provider.mark_success()
                    if provider_manager:
                        provider_manager.mark_provider_success(current_provider.name)
                    response = JSONResponse(
                        content=anthropic_response_obj.model_dump(exclude_unset=True)
                    )
                    return await _complete_request_and_cleanup(response)

        except Exception as e:
            # 获取HTTP状态码（如果可用）
            http_status_code = getattr(e, 'status_code', None) or (
                getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
            )

            # 使用provider_manager判断是否应该failover
            # 检查是否是streaming错误事件
            streaming_error_type = getattr(e, 'streaming_error_type', None)
            if streaming_error_type:
                error_type, should_failover = provider_manager.get_error_classification(e, http_status_code)
                # 对于streaming错误，需要检查错误类型是否应该failover
                if streaming_error_type in provider_manager.settings.get('failover_error_types', []):
                    should_failover = True
                else:
                    should_failover = False
            else:
                error_type, should_failover = provider_manager.get_error_classification(e, http_status_code)

            warning(
                LogRecord(
                    event="provider_request_failed",
                    message=f"Request failed for provider {current_provider.name} (attempt {attempt + 1}/{max_retries}): {str(e)}",
                    request_id=request_id,
                    data={
                        "provider": current_provider.name,
                        "target_model": target_model_name,
                        "attempt": attempt + 1,
                        "remaining_options": max_retries - attempt - 1,
                        "error_type": error_type,
                        "should_failover": should_failover,
                        "http_status_code": http_status_code
                    }
                ),
                exc=e
            )

            # 标记当前provider失败
            current_provider.mark_failure()

            # 如果不应该failover，直接返回错误给客户端
            if not should_failover:
                info(
                    LogRecord(
                        event="error_not_retryable",
                        message=f"Error type '{error_type}' not configured for failover, returning to client",
                        request_id=request_id,
                        data={
                            "provider": current_provider.name,
                            "error_type": error_type,
                            "http_status_code": http_status_code
                        }
                    )
                )

                # 根据错误类型选择合适的Anthropic错误类型
                if http_status_code == 400:
                    anthropic_error_type = AnthropicErrorType.INVALID_REQUEST
                elif http_status_code == 401:
                    anthropic_error_type = AnthropicErrorType.AUTHENTICATION
                elif http_status_code == 403:
                    anthropic_error_type = AnthropicErrorType.PERMISSION
                elif http_status_code == 404:
                    anthropic_error_type = AnthropicErrorType.NOT_FOUND
                elif http_status_code == 429:
                    anthropic_error_type = AnthropicErrorType.RATE_LIMIT
                else:
                    anthropic_error_type = AnthropicErrorType.API_ERROR

                response = await _log_and_return_error_response(
                    request,
                    http_status_code or 500,
                    anthropic_error_type,
                    str(e),
                    caught_exception=e,
                )
                _complete_and_cleanup_request(request_signature, response)
                return response

            # 如果还有其他选项，继续尝试下一个
            if attempt < max_retries - 1:
                next_target_model, next_provider = provider_options[attempt + 1]
                info(
                    LogRecord(
                        event="provider_fallback",
                        message=f"Falling back to provider: {next_provider.name} with model: {next_target_model}",
                        request_id=request_id,
                        data={
                            "failed_provider": current_provider.name,
                            "failed_model": target_model_name,
                            "fallback_provider": next_provider.name,
                            "fallback_model": next_target_model
                        }
                    )
                )
            # 如果这是最后一个选项，循环会自然结束

    # All providers failed
    response = await _log_and_return_error_response(
        request,
        503,
        AnthropicErrorType.API_ERROR,
        "All providers are currently unavailable. Please try again later.",
        caught_exception=None,
    )
    _complete_and_cleanup_request(request_signature, response)
    return response


@app.post(
    "/v1/messages/count_tokens", response_model=TokenCountResponse, tags=["Utility"]
)
async def count_tokens_endpoint(request: Request) -> TokenCountResponse:
    """Estimates token count for given Anthropic messages and system prompt."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time_mono = time.monotonic()

    try:
        body = await request.json()
        count_request = TokenCountRequest.model_validate(body)
    except json.JSONDecodeError as e:
        raise fastapi.HTTPException(status_code=400, detail="Invalid JSON body.") from e
    except ValidationError as e:
        raise fastapi.HTTPException(
            status_code=422, detail=f"Invalid request body: {e.errors()}"
        ) from e

    token_count = count_tokens_for_anthropic_request(
        messages=count_request.messages,
        system=count_request.system,
        model_name=count_request.model,
        tools=count_request.tools,
        request_id=request_id,
    )
    duration_ms = (time.monotonic() - start_time_mono) * 1000
    info(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Counted {token_count} tokens",
            request_id=request_id,
            data={
                "duration_ms": duration_ms,
                "token_count": token_count,
                "model": count_request.model,
            },
        )
    )
    return TokenCountResponse(input_tokens=token_count)


@app.get("/", include_in_schema=False, tags=["Health"])
async def root_health_check() -> JSONResponse:
    """Basic health check and information endpoint."""
    debug(
        LogRecord(
            event=LogEvent.HEALTH_CHECK.value, message="Root health check accessed"
        )
    )
    return JSONResponse(
        {
            "proxy_name": settings.app_name,
            "version": settings.app_version,
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.get("/providers", tags=["Health"])
async def get_providers_status() -> JSONResponse:
    """Get status of all configured providers."""
    if not provider_manager:
        return JSONResponse(
            {
                "error": "Provider manager not initialized",
                "status": "error"
            },
            status_code=500
        )

    status = provider_manager.get_status()
    return JSONResponse(status)


@app.post("/providers/reload", tags=["Health"])
async def reload_providers_config() -> JSONResponse:
    """Reload providers configuration from file."""
    if not provider_manager:
        return JSONResponse(
            {
                "error": "Provider manager not initialized",
                "status": "error"
            },
            status_code=500
        )

    try:
        provider_manager.reload_config()
        return JSONResponse(
            {
                "message": "Provider configuration reloaded successfully",
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        return JSONResponse(
            {
                "error": f"Failed to reload configuration: {str(e)}",
                "status": "error"
            },
            status_code=500
        )


@app.exception_handler(openai.APIError)
async def openai_api_error_handler(request: Request, exc: openai.APIError):
    # Recursion protection
    if not hasattr(_exception_handler_depth, 'value'):
        _exception_handler_depth.value = 0

    with _exception_handler_lock:
        if _exception_handler_depth.value >= 3:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "API error - recursion protection activated"
                    }
                },
                status_code=500
            )

        _exception_handler_depth.value += 1
        try:
            err_type, err_msg, err_status, prov_details = _get_anthropic_error_details_from_exc(
                exc
            )
            return await _log_and_return_error_response(
                request, err_status, err_type, err_msg, prov_details, exc
            )
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "API error handling failed"
                    }
                },
                status_code=500
            )
        finally:
            _exception_handler_depth.value -= 1


@app.exception_handler(ValidationError)
async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
    # Recursion protection
    if not hasattr(_exception_handler_depth, 'value'):
        _exception_handler_depth.value = 0

    with _exception_handler_lock:
        if _exception_handler_depth.value >= 3:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Validation error - recursion protection activated"
                    }
                },
                status_code=422
            )

        _exception_handler_depth.value += 1
        try:
            return await _log_and_return_error_response(
                request,
                422,
                AnthropicErrorType.INVALID_REQUEST,
                f"Validation error: {exc.errors()}",
                caught_exception=exc,
            )
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Validation error handling failed"
                    }
                },
                status_code=422
            )
        finally:
            _exception_handler_depth.value -= 1


@app.exception_handler(json.JSONDecodeError)
async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
    # Recursion protection
    if not hasattr(_exception_handler_depth, 'value'):
        _exception_handler_depth.value = 0

    with _exception_handler_lock:
        if _exception_handler_depth.value >= 3:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "JSON decode error - recursion protection activated"
                    }
                },
                status_code=400
            )

        _exception_handler_depth.value += 1
        try:
            return await _log_and_return_error_response(
                request,
                400,
                AnthropicErrorType.INVALID_REQUEST,
                "Invalid JSON format.",
                caught_exception=exc,
            )
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "JSON decode error handling failed"
                    }
                },
                status_code=400
            )
        finally:
            _exception_handler_depth.value -= 1


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Recursion protection
    if not hasattr(_exception_handler_depth, 'value'):
        _exception_handler_depth.value = 0

    with _exception_handler_lock:
        if _exception_handler_depth.value >= 3:  # Allow max 3 levels of recursion
            # Emergency fallback - return minimal response without logging
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Internal server error - recursion protection activated"
                    }
                },
                status_code=500
            )

        _exception_handler_depth.value += 1
        try:
            return await _log_and_return_error_response(
                request,
                500,
                AnthropicErrorType.API_ERROR,
                "An unexpected internal server error occurred.",
                caught_exception=exc,
            )
        except Exception:
            # If error handling itself fails, return minimal response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Internal server error - error handling failed"
                    }
                },
                status_code=500
            )
        finally:
            _exception_handler_depth.value -= 1


@app.middleware("http")
async def logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    if not hasattr(request.state, "start_time_monotonic"):
        request.state.start_time_monotonic = time.monotonic()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request.state.request_id
    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
    response.headers["X-Response-Time-ms"] = str(duration_ms)

    return response


if __name__ == "__main__":
    if provider_manager:
        # Display provider information
        providers_text = ""
        healthy_count = len(provider_manager.get_healthy_providers())
        total_count = len(provider_manager.providers)

        for i, provider in enumerate(provider_manager.providers):
            status_icon = "✓" if provider.is_healthy(provider_manager.get_failure_cooldown()) else "✗"
            provider_line = f"\n   [{status_icon}] {provider.name} ({provider.type.value}): {provider.base_url}"
            providers_text += provider_line

        config_details_text = Text.assemble(
          ("   Version       : ", "default"),
          (f"v{settings.app_version}", "bold cyan"),
          ("\n   Providers     : ", "default"),
          (f"{healthy_count}/{total_count} healthy", "bold green" if healthy_count > 0 else "bold red"),
          (providers_text, "default"),
          ("\n   Log Level     : ", "default"),
          (settings.log_level.upper(), "yellow"),
          ("\n   Log File      : ", "default"),
          (settings.log_file_path or "Disabled", "dim"),
          ("\n   Listening on  : ", "default"),
          (f"http://{settings.host}:{settings.port}", "bold white"),
          ("\n   Reload        : ", "default"),
          ("Enabled", "bold orange1") if settings.reload else ("Disabled", "dim")
        )

        title = "Claude Code Provider Balancer Configuration"
    else:
        config_details_text = Text.assemble(
          ("   Version       : ", "default"),
          (f"v{settings.app_version}", "bold cyan"),
          ("\n   Status        : ", "default"),
          ("Provider manager failed to initialize", "bold red"),
          ("\n   Log Level     : ", "default"),
          (settings.log_level.upper(), "yellow"),
          ("\n   Listening on  : ", "default"),
          (f"http://{settings.host}:{settings.port}", "bold white"),
        )
        title = "Claude Code Provider Balancer Configuration (ERROR)"

    _console.print(
      Panel(
        config_details_text,
        title=title,
        border_style="blue",
        expand=False,
      )
    )
    _console.print(Rule("Starting Uvicorn server...", style="dim blue"))

    # Setup signal handlers for graceful shutdown
    import signal

    def shutdown_handler(signum, frame):
        """Handle shutdown signals"""
        from rich.console import Console
        console = Console()
        console.print("\n[yellow]📴 Shutting down gracefully...[/yellow]")
        if provider_manager:
            provider_manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Configure reload directories and patterns if reload is enabled
        reload_dirs = None
        reload_includes = None
        if settings.reload:
            # Include both Python files and the providers.yaml config file
            reload_dirs = [str(Path(__file__).parent.parent)]  # Project root directory
            reload_includes = ["*.py", "providers.yaml"]

        uvicorn.run(
            "__main__:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            reload_dirs=reload_dirs,
            reload_includes=reload_includes,
            log_config=log_config,
            access_log=False,
        )
    finally:
        # Ensure cleanup on exit
        if provider_manager:
            provider_manager.shutdown()
