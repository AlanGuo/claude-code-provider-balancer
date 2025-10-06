"""Logging handlers and utility functions."""

import enum
import logging
import traceback
from typing import Optional

from .formatters import LogError, LogRecord


class LogEvent(enum.Enum):
    # Core request flow events
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILURE = "request_failure"
    REQUEST_RECEIVED = "request_received"
    
    # Provider communication events
    PROVIDER_REQUEST = "provider_request"
    PROVIDER_REQUEST_ERROR = "provider_request_error"
    PROVIDER_HTTP_ERROR_DETAILS = "provider_http_error_details"
    PROVIDER_API_ERROR_DETAILS = "provider_api_error_details"
    PROVIDER_LOADED = "provider_loaded"
    PROVIDER_FALLBACK = "provider_fallback"
    ALL_PROVIDERS_FAILED = "all_providers_failed"
    PROVIDER_ERROR_BELOW_THRESHOLD = "provider_error_below_threshold"  # Provider错误数未达阈值
    PROVIDER_UNHEALTHY_NO_FAILOVER = "provider_unhealthy_no_failover"  # Provider不健康但无法failover
    GET_PROVIDER_HEADERS_START = "get_provider_headers_start"
    ORIGINAL_REQUEST_HEADERS_RECEIVED = "original_request_headers_received"
    FINAL_PROVIDER_HEADERS = "final_provider_headers"
    CLAUDE_OFFICIAL_HEADERS_APPLIED = "claude_official_headers_applied"

    # Streaming events
    SSE_EXTRACTION_COMPLETE = "sse_extraction_complete"
    ERROR_SENT_TO_CLIENT = "error_sent_to_client"
    FAILED_TO_SEND_ERROR_TO_CLIENT = "failed_to_send_error_to_client"
    PROVIDER_STREAM_ERROR = "provider_stream_error"
    CHUNK_AVAILABLE_FOR_DUPLICATES = "chunk_available_for_duplicates"
    
    # Client streaming events
    CHUNK_PREPARED_FOR_CLIENT = "chunk_prepared_for_client"
    CLIENT_DISCONNECTED_DURING_SEND = "client_disconnected_during_send"
    CLIENT_ADDED_TO_BROADCASTER = "client_added_to_broadcaster"
    DUPLICATE_REQUEST_MID_STREAM = "duplicate_request_mid_stream"
    HISTORICAL_CHUNK_YIELDED_TO_DUPLICATE = "historical_chunk_yielded_to_duplicate"
    DUPLICATE_DISCONNECTED_DURING_HISTORICAL_CHUNK = "duplicate_disconnected_during_historical_chunk"
    LIVE_CHUNK_YIELDED_TO_DUPLICATE = "live_chunk_yielded_to_duplicate"
    DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK = "duplicate_disconnected_during_live_chunk"
    NO_ACTIVE_CLIENTS_FOR_BROADCAST = "no_active_clients_for_broadcast"
    BROADCAST_CHUNK_COMPLETED = "broadcast_chunk_completed"
    PARALLEL_BROADCAST_STARTED = "parallel_broadcast_started"
    CHUNK_YIELDED_TO_ORIGINAL_CLIENT = "chunk_yielded_to_original_client"
    ORIGINAL_CLIENT_DISCONNECTED_DURING_YIELD = "original_client_disconnected_during_yield"
    CONTINUING_FOR_DUPLICATE_CLIENTS = "continuing_for_duplicate_clients"
    STOPPING_NO_DUPLICATE_CLIENTS = "stopping_no_duplicate_clients"
    BROADCAST_SESSION_SUMMARY = "broadcast_session_summary"
    
    # Broadcaster registry events
    BROADCASTER_REGISTERED = "broadcaster_registered"
    BROADCASTER_UNREGISTERED = "broadcaster_unregistered"
    DUPLICATE_REQUEST_FOUND_ACTIVE_BROADCASTER = "duplicate_request_found_active_broadcaster"
    DUPLICATE_REQUEST_NO_ACTIVE_BROADCASTER = "duplicate_request_no_active_broadcaster"
    
    # Token counting events
    TOKEN_COUNT = "token_count"
    TOKEN_ENCODER_LOAD_FAILED = "token_encoder_load_failed"
    COUNT_TOKENS_API_CALL = "count_tokens_api_call"
    COUNT_TOKENS_FALLBACK = "count_tokens_fallback"
    
    # Message processing events
    SYSTEM_PROMPT_ADJUSTED = "system_prompt_adjusted"
    
    # Tool processing events
    TOOL_INPUT_SERIALIZATION_FAILURE = "tool_input_serialization_failure"
    TOOL_RESULT_SERIALIZATION_FAILURE = "tool_result_serialization_failure"
    TOOL_RESULT_PROCESSING = "tool_result_processing"
    TOOL_CHOICE_UNSUPPORTED = "tool_choice_unsupported"
    TOOL_ARGS_PARSE_FAILURE = "tool_args_parse_failure"
    
    # System events
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    FASTAPI_STARTUP_COMPLETE = "fastapi_startup_complete"
    FASTAPI_STARTUP_FAILED = "fastapi_startup_failed"
    FASTAPI_SHUTDOWN = "fastapi_shutdown"
    HTTP_REQUEST = "http_request"
    
    # Request deduplication and cleanup events
    SIGNATURE_GENERATED = "signature_generated"
    DUPLICATE_REQUEST_RECEIVED = "duplicate_request_received"
    REQUEST_COMPLETED_DELAY_CLEANUP = "request_completed_delay_cleanup"
    DUPLICATE_REQUESTS_COMPLETED_DELAY_CLEANUP = "duplicate_requests_completed_delay_cleanup"
    NO_DUPLICATE_REQUESTS_FOUND = "no_duplicate_requests_found"
    LATE_DUPLICATE_REQUESTS_FOUND = "late_duplicate_requests_found"
    LATE_DUPLICATE_FUTURE_SET_RESULT = "late_duplicate_future_set_result"
    LATE_DUPLICATE_COMPLETION_SUMMARY = "late_duplicate_completion_summary"
    DELAYED_CLEANUP_COMPLETED = "delayed_cleanup_completed"
    ORIGINAL_REQUEST_COMPLETED = "original_request_completed"
    STREAMING_COMPLETED = "streaming_completed"
    REQUEST_CLEANUP = "request_cleanup"
    REQUEST_CLEANUP_SKIP = "request_cleanup_skip"
    STUCK_REQUEST_CLEANUP = "stuck_request_cleanup"
    
    # Provider health check events
    PROVIDER_UNHEALTHY_NON_STREAM = "provider_unhealthy_non_stream"
    PROVIDER_HEALTH_ERROR_RECORDED = "provider_health_error_recorded"
    PROVIDER_MARKED_UNHEALTHY = "provider_marked_unhealthy"
    PROVIDER_HEALTH_ERROR_COUNT_RESET = "provider_health_error_count_reset"
    PROVIDER_HEALTH_ERROR_COUNT_TIMEOUT_RESET = "provider_health_error_count_timeout_reset"

    # OAuth events
    OAUTH_KEYRING_UNAVAILABLE = "oauth_keyring_unavailable"
    OAUTH_TOKENS_SAVED = "oauth_tokens_saved"
    OAUTH_SAVE_FAILED = "oauth_save_failed"
    OAUTH_TOKENS_SAVED_ASYNC = "oauth_tokens_saved_async"
    OAUTH_SAVE_FAILED_ASYNC = "oauth_save_failed_async"
    OAUTH_NO_TOKENS_FOUND = "oauth_no_tokens_found"
    OAUTH_TOKENS_LOADED = "oauth_tokens_loaded"
    OAUTH_LOAD_FAILED = "oauth_load_failed"
    OAUTH_TOKENS_CLEANED = "oauth_tokens_cleaned"
    OAUTH_TOKENS_SAVED_AFTER_CLEANUP = "oauth_tokens_saved_after_cleanup"
    OAUTH_SAVE_FAILED_AFTER_CLEANUP = "oauth_save_failed_after_cleanup"
    OAUTH_URL_GENERATED = "oauth_url_generated"
    OAUTH_URL_DEBUG = "oauth_url_debug"
    OAUTH_NO_STATE = "oauth_no_state"
    OAUTH_STATE_EXPIRED = "oauth_state_expired"
    OAUTH_TOKEN_EXCHANGE_FAILED = "oauth_token_exchange_failed"
    OAUTH_DUPLICATE_FOUND_EMAIL = "oauth_duplicate_found_email"
    OAUTH_DUPLICATE_FOUND_TOKEN = "oauth_duplicate_found_token"
    OAUTH_DUPLICATE_FOUND_REFRESH = "oauth_duplicate_found_refresh"
    OAUTH_DUPLICATE_FOUND_FINGERPRINT = "oauth_duplicate_found_fingerprint"
    OAUTH_DUPLICATE_FOUND_ACCOUNT = "oauth_duplicate_found_account"
    OAUTH_TOKEN_REMOVED = "oauth_token_removed"
    OAUTH_TOKEN_ADDED = "oauth_token_added"
    OAUTH_TOKENS_SAVED_AFTER_EXCHANGE = "oauth_tokens_saved_after_exchange"
    OAUTH_SAVE_FAILED_AFTER_EXCHANGE = "oauth_save_failed_after_exchange"
    OAUTH_EXCHANGE_SUCCESS = "oauth_exchange_success"
    OAUTH_TOKEN_EXCHANGE_DEBUG = "oauth_token_exchange_debug"
    OAUTH_TOKEN_EXPIRY = "oauth_token_expiry"
    OAUTH_AUTO_REFRESH_STARTED = "oauth_auto_refresh_started"
    OAUTH_EXCHANGE_ERROR = "oauth_exchange_error"
    OAUTH_REFRESH_REQUEST = "oauth_refresh_request"
    OAUTH_REFRESH_RESPONSE_STATUS = "oauth_refresh_response_status"
    OAUTH_REFRESH_RESPONSE_TEXT = "oauth_refresh_response_text"
    OAUTH_REFRESH_FAILED = "oauth_refresh_failed"
    OAUTH_REFRESH_SUCCESSFUL = "oauth_refresh_successful"
    OAUTH_TOKEN_REFRESHED = "oauth_token_refreshed"
    OAUTH_NEW_TOKEN_EXPIRY = "oauth_new_token_expiry"
    OAUTH_REFRESH_ERROR = "oauth_refresh_error"
    OAUTH_TOKEN_USED = "oauth_token_used"
    OAUTH_ALL_TOKENS_INVALID = "oauth_all_tokens_invalid"
    OAUTH_TOKEN_NEEDS_REFRESH = "oauth_token_needs_refresh"
    OAUTH_AUTO_REFRESH_FAILED = "oauth_auto_refresh_failed"
    OAUTH_NEXT_REFRESH_SCHEDULED = "oauth_next_refresh_scheduled"
    OAUTH_AUTO_REFRESH_CANCELLED = "oauth_auto_refresh_cancelled"
    OAUTH_AUTO_REFRESH_LOOP_ERROR = "oauth_auto_refresh_loop_error"
    OAUTH_REFRESH_BY_EMAIL_NOT_FOUND = "oauth_refresh_by_email_not_found"
    OAUTH_REFRESH_BY_EMAIL_NO_REFRESH_TOKEN = "oauth_refresh_by_email_no_refresh_token"
    OAUTH_REFRESH_BY_EMAIL_START = "oauth_refresh_by_email_start"
    OAUTH_REFRESH_BY_EMAIL_SUCCESS = "oauth_refresh_by_email_success"
    OAUTH_REFRESH_BY_EMAIL_FAILED = "oauth_refresh_by_email_failed"
    OAUTH_TOKEN_REMOVED_BY_EMAIL = "oauth_token_removed_by_email"
    OAUTH_TOKENS_SAVED_AFTER_REMOVAL = "oauth_tokens_saved_after_removal"
    OAUTH_SAVE_FAILED_AFTER_REMOVAL = "oauth_save_failed_after_removal"
    OAUTH_ALL_TOKENS_CLEARED = "oauth_all_tokens_cleared"
    OAUTH_TOKENS_SAVED_AFTER_CLEAR = "oauth_tokens_saved_after_clear"
    OAUTH_SAVE_FAILED_AFTER_CLEAR = "oauth_save_failed_after_clear"
    OAUTH_EXISTING_TOKENS_FOUND = "oauth_existing_tokens_found"
    OAUTH_MANAGER_INITIALIZED = "oauth_manager_initialized"
    OAUTH_AUTO_REFRESH_ALL_STARTED = "oauth_auto_refresh_all_started"
    OAUTH_AUTO_REFRESH_DISABLED = "oauth_auto_refresh_disabled"
    OAUTH_MANAGER_READY = "oauth_manager_ready"
    OAUTH_AUTO_REFRESH_START_FAILED = "oauth_auto_refresh_start_failed"
    OAUTH_URL_GENERATION_ERROR = "oauth_url_generation_error"
    OAUTH_MANUAL_REFRESH_ERROR = "oauth_manual_refresh_error"
    OAUTH_MANAGER_CHECK = "oauth_manager_check"
    OAUTH_MANAGER_INIT_FAILED = "oauth_manager_init_failed"
    OAUTH_GET_TOKEN_BY_EMAIL_NO_EMAIL = "oauth_get_token_by_email_no_email"
    OAUTH_GET_TOKEN_BY_EMAIL_NO_TOKENS = "oauth_get_token_by_email_no_tokens"
    OAUTH_TOKEN_USED_BY_EMAIL = "oauth_token_used_by_email"
    OAUTH_TOKEN_EXPIRED_BY_EMAIL = "oauth_token_expired_by_email"
    OAUTH_ACCOUNT_NOT_FOUND = "oauth_account_not_found"


# Initialize logger - will be set up when module is initialized
_logger = None


def init_logger(app_name: str = "claude-provider-balancer"):
    """Initialize the logger for this module."""
    global _logger
    _logger = logging.getLogger(app_name)


def _log(level: int, record: LogRecord, exc: Optional[Exception] = None) -> None:
    """Internal logging function."""
    if _logger is None:
        init_logger()
    
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

        # Ensure the message is safe for logging by checking encoding
        safe_message = record.message
        try:
            safe_message.encode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback to ASCII representation for problematic characters
            safe_message = safe_message.encode('ascii', errors='replace').decode('ascii')
        
        _logger.log(level=level, msg=safe_message, extra={"log_record": record})
    except Exception:
        # Last resort: use standard Python logging without custom formatting
        try:
            import logging as std_logging
            std_logging.getLogger("fallback").log(level, f"Log error: {record.message}")
        except Exception:
            pass  # Silent failure to prevent infinite recursion


def debug(record: LogRecord):
    """Log a debug message."""
    _log(logging.DEBUG, record)


def info(record: LogRecord):
    """Log an info message."""
    _log(logging.INFO, record)


def warning(record: LogRecord, exc: Optional[Exception] = None):
    """Log a warning message."""
    _log(logging.WARNING, record, exc=exc)


def error(record: LogRecord, exc: Optional[Exception] = None):
    """Log an error message."""
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
    """Log a critical message."""
    _log(logging.CRITICAL, record, exc=exc)


def error_file_only(record: LogRecord, exc: Optional[Exception] = None):
    """Log an error message only to file, not to console."""
    try:
        if _logger is None:
            init_logger()
        
        # Create a copy of the logger with only file handler
        file_logger = logging.getLogger(f"{_logger.name}.file_only")
        
        # Remove all existing handlers to prevent console output
        file_logger.handlers.clear()
        file_logger.propagate = False
        
        # Add only file handler if it exists in the main logger
        for handler in _logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_logger.addHandler(handler)
                break
        
        # If no file handler found, try to get it from parent logger
        if not file_logger.handlers and _logger.parent:
            for handler in _logger.parent.handlers:
                if isinstance(handler, logging.FileHandler):
                    file_logger.addHandler(handler)
                    break
        
        # Process the log record similar to _log function
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

        # Ensure the message is safe for logging
        safe_message = record.message
        try:
            safe_message.encode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback to ASCII representation for problematic characters
            safe_message = safe_message.encode('ascii', errors='replace').decode('ascii')

        # Log only to file
        if file_logger.handlers:
            file_logger.error(safe_message, extra={"log_record": record})
        else:
            # Fallback: if no file handler available, log to main logger with a marker
            record.message = f"[FILE_ONLY] {record.message}"
            _log(logging.ERROR, record, exc=exc)
            
    except Exception:
        # Fallback to regular error logging
        error(record, exc=exc)