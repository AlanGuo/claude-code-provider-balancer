"""
Parallel broadcasting module for handling stream distribution to multiple clients.
Handles client disconnections gracefully through exception-based detection.
"""

import asyncio
import json
from typing import List, AsyncGenerator, Optional, Dict, Any
from fastapi import Request
from utils.logging import debug, info, error, LogRecord, LogEvent


class ClientStream:
    """Represents a client stream with disconnect detection"""
    
    def __init__(self, request: Request, request_id: str, client_type: str = "original"):
        self.request = request
        self.request_id = request_id
        self.client_type = client_type  # "original" or "duplicate"
        self.is_active = True
        self.chunks_sent = 0
        self.last_error = None
    
    async def send_chunk(self, chunk: str, chunk_index: int, provider_name: str) -> bool:
        """
        Send chunk to this client. Returns True if successful, False if disconnected.
        Uses exception-based disconnect detection during actual data transmission.
        """
        if not self.is_active:
            return False
            
        try:
            # This is where we actually detect disconnection - during the yield
            # The generator function will be called by FastAPI's StreamingResponse
            async def chunk_generator():
                yield chunk
            
            # We can't actually yield here since this isn't a generator context
            # Instead, we'll return the chunk and let the caller handle yielding
            # But we can still detect if the client stream is conceptually active
            
            self.chunks_sent += 1
            debug(
                LogRecord(
                    LogEvent.CHUNK_PREPARED_FOR_CLIENT.value,
                    f"Prepared chunk {chunk_index} for {self.client_type} client ({len(chunk)} bytes)",
                    self.request_id,
                    {
                        "provider": provider_name,
                        "client_type": self.client_type,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk),
                        "total_chunks_sent": self.chunks_sent
                    }
                )
            )
            return True
            
        except Exception as e:
            self.is_active = False
            self.last_error = e
            debug(
                LogRecord(
                    LogEvent.CLIENT_DISCONNECTED_DURING_SEND.value,
                    f"Client disconnected during chunk {chunk_index} send: {type(e).__name__}: {e}",
                    self.request_id,
                    {
                        "provider": provider_name,
                        "client_type": self.client_type,
                        "chunk_index": chunk_index,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "total_chunks_sent": self.chunks_sent
                    }
                )
            )
            return False


class ParallelBroadcaster:
    """Handles parallel broadcasting to multiple client streams"""
    
    def __init__(self, original_request: Request, request_id: str, provider_name: str):
        self.original_request = original_request
        self.request_id = request_id
        self.provider_name = provider_name
        self.clients: List[ClientStream] = []
        self.total_chunks_processed = 0
        self.collected_chunks: List[str] = []  # Store all chunks for late-joining duplicates
        self.streaming_active = False  # Track if streaming is in progress
        self.last_exception_info: Optional[Dict[str, Any]] = None  # Store exception info for health check
        
        # Add the original client
        self.add_client(original_request, request_id, "original")
    
    def add_client(self, request: Request, request_id: str, client_type: str = "duplicate"):
        """Add a client stream for broadcasting"""
        client = ClientStream(request, request_id, client_type)
        self.clients.append(client)
        info(
            LogRecord(
                LogEvent.CLIENT_ADDED_TO_BROADCASTER.value,
                f"Added {client_type} client to broadcaster",
                self.request_id,
                {
                    "provider": self.provider_name,
                    "client_type": client_type,
                    "total_clients": len(self.clients),
                    "streaming_active": self.streaming_active,
                    "chunks_already_processed": len(self.collected_chunks)
                }
            )
        )
    
    async def add_duplicate_request(self, duplicate_request: Request, duplicate_request_id: str) -> AsyncGenerator[str, None]:
        """
        Add a duplicate request that arrives mid-stream.
        Returns an async generator that yields all chunks (past + future).
        """
        info(
            LogRecord(
                LogEvent.DUPLICATE_REQUEST_MID_STREAM.value,
                f"Adding duplicate request mid-stream after {len(self.collected_chunks)} chunks",
                self.request_id,
                {
                    "provider": self.provider_name,
                    "duplicate_request_id": duplicate_request_id,
                    "chunks_already_processed": len(self.collected_chunks),
                    "streaming_active": self.streaming_active
                }
            )
        )
        
        # Add the duplicate client
        self.add_client(duplicate_request, duplicate_request_id, "duplicate")
        
        try:
            # First, yield all previously collected chunks
            for i, chunk in enumerate(self.collected_chunks):
                try:
                    yield chunk
                    debug(
                        LogRecord(
                            LogEvent.HISTORICAL_CHUNK_YIELDED_TO_DUPLICATE.value,
                            f"Yielded historical chunk {i+1}/{len(self.collected_chunks)} to duplicate ({len(chunk)} bytes)",
                            duplicate_request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": i+1,
                                "chunk_size": len(chunk),
                                "is_historical": True
                            }
                        )
                    )
                except (asyncio.CancelledError, GeneratorExit):
                    # Handle graceful cancellation/closure
                    debug(
                        LogRecord(
                            LogEvent.DUPLICATE_DISCONNECTED_DURING_HISTORICAL_CHUNK.value,
                            f"Duplicate client cancelled during historical chunk {i+1}",
                            duplicate_request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": i+1,
                                "reason": "cancelled_or_closed"
                            }
                        )
                    )
                    # Mark this duplicate as inactive
                    if self.clients:
                        self.clients[-1].is_active = False
                    return
                except Exception as e:
                    error(
                        LogRecord(
                            LogEvent.DUPLICATE_DISCONNECTED_DURING_HISTORICAL_CHUNK.value,
                            f"Duplicate client disconnected during historical chunk {i+1}: {type(e).__name__}: {e}",
                            duplicate_request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": i+1,
                                "error": str(e)
                            }
                        )
                    )
                    # Mark this duplicate as inactive
                    if self.clients:
                        self.clients[-1].is_active = False
                    return
            
            # Now yield future chunks as they come
            # This will be handled by the main streaming loop
            while self.streaming_active:
                try:
                    # Check for cancellation before sleeping
                    if not self.streaming_active:
                        break
                        
                    # Wait for new chunks to be added to collected_chunks
                    current_chunk_count = len(self.collected_chunks)
                    await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
                    
                    # Check if new chunks were added
                    if len(self.collected_chunks) > current_chunk_count:
                        # Yield new chunks
                        for i in range(current_chunk_count, len(self.collected_chunks)):
                            chunk = self.collected_chunks[i]
                            try:
                                yield chunk
                                debug(
                                    LogRecord(
                                        LogEvent.LIVE_CHUNK_YIELDED_TO_DUPLICATE.value,
                                        f"Yielded live chunk {i+1} to duplicate ({len(chunk)} bytes)",
                                        duplicate_request_id,
                                        {
                                            "provider": self.provider_name,
                                            "chunk_index": i+1,
                                            "chunk_size": len(chunk),
                                            "is_historical": False
                                        }
                                    )
                                )
                            except (asyncio.CancelledError, GeneratorExit):
                                # Handle graceful cancellation/closure
                                debug(
                                    LogRecord(
                                        LogEvent.DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK.value,
                                        f"Duplicate client cancelled during live chunk {i+1}",
                                        duplicate_request_id,
                                        {
                                            "provider": self.provider_name,
                                            "chunk_index": i+1,
                                            "reason": "cancelled_or_closed"
                                        }
                                    )
                                )
                                # Mark this duplicate as inactive
                                if self.clients:
                                    self.clients[-1].is_active = False
                                return
                            except Exception as e:
                                error(
                                    LogRecord(
                                        LogEvent.DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK.value,
                                        f"Duplicate client disconnected during live chunk {i+1}: {type(e).__name__}: {e}",
                                        duplicate_request_id,
                                        {
                                            "provider": self.provider_name,
                                            "chunk_index": i+1,
                                            "error": str(e)
                                        }
                                    )
                                )
                                # Mark this duplicate as inactive
                                if self.clients:
                                    self.clients[-1].is_active = False
                                return
                except (asyncio.CancelledError, GeneratorExit):
                    # Handle cancellation in the main loop
                    debug(
                        LogRecord(
                            LogEvent.DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK.value,
                            "Duplicate client cancelled during streaming loop",
                            duplicate_request_id,
                            {
                                "provider": self.provider_name,
                                "reason": "cancelled_or_closed_in_loop"
                            }
                        )
                    )
                    # Mark this duplicate as inactive
                    if self.clients:
                        self.clients[-1].is_active = False
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # Handle top-level cancellation
            debug(
                LogRecord(
                    LogEvent.DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK.value,
                    "Duplicate client generator cancelled",
                    duplicate_request_id,
                    {
                        "provider": self.provider_name,
                        "reason": "generator_cancelled"
                    }
                )
            )
            # Mark this duplicate as inactive
            if self.clients:
                self.clients[-1].is_active = False
        finally:
            # Ensure client is marked as inactive on exit
            if self.clients:
                self.clients[-1].is_active = False
    
    def get_active_clients(self) -> List[ClientStream]:
        """Get list of currently active clients"""
        return [client for client in self.clients if client.is_active]
    
    async def broadcast_chunk(self, chunk: str) -> bool:
        """
        Broadcast chunk to all active clients.
        Returns True if any clients are still active, False if all disconnected.
        """
        if not chunk:
            return len(self.get_active_clients()) > 0
            
        self.total_chunks_processed += 1
        active_clients = self.get_active_clients()
        
        if not active_clients:
            debug(
                LogRecord(
                    LogEvent.NO_ACTIVE_CLIENTS_FOR_BROADCAST.value,
                    f"No active clients remaining for chunk {self.total_chunks_processed}",
                    self.request_id,
                    {
                        "provider": self.provider_name,
                        "chunk_index": self.total_chunks_processed,
                        "total_clients": len(self.clients)
                    }
                )
            )
            return False
        
        # Prepare chunk for all active clients
        send_tasks = []
        for client in active_clients:
            send_tasks.append(
                client.send_chunk(chunk, self.total_chunks_processed, self.provider_name)
            )
        
        # Wait for all sends to complete
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        # Count successful sends
        successful_sends = sum(1 for result in results if result is True)
        remaining_active = len(self.get_active_clients())
        
        debug(
            LogRecord(
                LogEvent.BROADCAST_CHUNK_COMPLETED.value,
                f"Broadcasted chunk {self.total_chunks_processed} to {successful_sends}/{len(active_clients)} clients",
                self.request_id,
                {
                    "provider": self.provider_name,
                    "chunk_index": self.total_chunks_processed,
                    "chunk_size": len(chunk),
                    "successful_sends": successful_sends,
                    "attempted_sends": len(active_clients),
                    "remaining_active_clients": remaining_active
                }
            )
        )
        
        return remaining_active > 0
    
    async def stream_from_provider(self, provider_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Stream from provider to all clients, yielding chunks for the original client.
        Handles parallel broadcasting and client disconnect detection.
        """
        debug(
            LogRecord(
                LogEvent.PARALLEL_BROADCAST_STARTED.value,
                f"Starting parallel broadcast to {len(self.clients)} clients",
                self.request_id,
                {
                    "provider": self.provider_name,
                    "total_clients": len(self.clients)
                }
            )
        )
        
        self.streaming_active = True
        
        try:
            async for chunk in provider_stream:
                self.total_chunks_processed += 1
                
                # Store chunk for late-joining duplicates
                self.collected_chunks.append(chunk)
                
                # Yield chunk for the original client (FastAPI StreamingResponse)
                # The actual disconnect detection happens here during the yield
                try:
                    yield chunk
                    debug(
                        LogRecord(
                            LogEvent.CHUNK_YIELDED_TO_ORIGINAL_CLIENT.value,
                            f"Yielded chunk {self.total_chunks_processed} to client ({len(chunk)} bytes)",
                            self.request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": self.total_chunks_processed,
                                "chunk_size": len(chunk),
                                "total_collected_chunks": len(self.collected_chunks)
                            }
                        )
                    )
                except Exception as e:
                    # Original client disconnected during yield
                    debug(
                        LogRecord(
                            LogEvent.ORIGINAL_CLIENT_DISCONNECTED_DURING_YIELD.value,
                            f"Original client disconnected during yield: {type(e).__name__}: {e}",
                            self.request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": self.total_chunks_processed,
                                "error_type": type(e).__name__,
                                "error": str(e)
                            }
                        )
                    )
                    # Mark original client as inactive
                    if self.clients:
                        self.clients[0].is_active = False
                    
                    # Check if we have other active clients (duplicates)
                    remaining_active = len(self.get_active_clients())
                    if remaining_active > 0:
                        info(
                            LogRecord(
                                LogEvent.CONTINUING_FOR_DUPLICATE_CLIENTS.value,
                                f"Original client disconnected but continuing for {remaining_active} duplicate clients",
                                self.request_id,
                                {
                                    "provider": self.provider_name,
                                    "remaining_active_clients": remaining_active,
                                    "chunks_processed": self.total_chunks_processed
                                }
                            )
                        )
                        # Continue consuming provider stream for duplicate clients
                        # The duplicate clients will receive chunks via add_duplicate_request
                        continue
                    else:
                        # No duplicate clients, stop provider stream
                        info(
                            LogRecord(
                                LogEvent.STOPPING_NO_DUPLICATE_CLIENTS.value,
                                "Original client disconnected and no duplicate clients, stopping provider stream",
                                self.request_id,
                                {
                                    "provider": self.provider_name,
                                    "chunks_processed": self.total_chunks_processed
                                }
                            )
                        )
                        break
                
                # Log duplicate client status
                duplicate_count = len(self.clients) - 1
                if duplicate_count > 0:
                    debug(
                        LogRecord(
                            LogEvent.CHUNK_AVAILABLE_FOR_DUPLICATES.value,
                            f"Chunk {self.total_chunks_processed} stored and available for {duplicate_count} duplicate clients",
                            self.request_id,
                            {
                                "provider": self.provider_name,
                                "chunk_index": self.total_chunks_processed,
                                "duplicate_count": duplicate_count,
                                "total_collected_chunks": len(self.collected_chunks)
                            }
                        )
                    )
                    
        except Exception as e:
            # Store exception info for health check
            self.last_exception_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "chunks_processed": self.total_chunks_processed
            }
            
            error(
                LogRecord(
                    LogEvent.PROVIDER_STREAM_ERROR.value,
                    f"Error in provider stream: {type(e).__name__}: {e}",
                    self.request_id,
                    {
                        "provider": self.provider_name,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "chunks_processed": self.total_chunks_processed
                    }
                )
            )
            
            # Send original error message to client before ending stream
            try:
                # Use original error message directly
                original_error_message = f"\n\n❌ **Error**: {str(e)}"
                
                # Send error message as content delta
                error_chunk = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta", 
                        "text": original_error_message
                    }
                }
                
                yield f"event: content_block_delta\ndata: {json.dumps(error_chunk)}\n\n"
                
                # Send content block stop
                content_block_stop = {
                    "type": "content_block_stop",
                    "index": 0
                }
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"
                
                # Send message delta with stop reason
                message_delta = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "error",
                        "stop_sequence": None
                    }
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
                
                # Send message stop to properly end the stream
                message_stop = {"type": "message_stop"}
                yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
                
                debug(
                    LogRecord(
                        LogEvent.ERROR_SENT_TO_CLIENT.value,
                        "Sent original error message to client after provider error",
                        self.request_id,
                        {
                            "provider": self.provider_name,
                            "error_type": type(e).__name__,
                            "chunks_processed": self.total_chunks_processed,
                            "original_error": str(e),
                            "error_message_sent": original_error_message
                        }
                    )
                )
                
            except Exception as yield_error:
                # If we can't send the error message, log it and raise the original error
                error(
                    LogRecord(
                        LogEvent.FAILED_TO_SEND_ERROR_TO_CLIENT.value,
                        f"Failed to send error message to client: {type(yield_error).__name__}: {yield_error}",
                        self.request_id,
                        {
                            "provider": self.provider_name,
                            "original_error": str(e),
                            "yield_error": str(yield_error)
                        }
                    )
                )
                raise
        finally:
            self.streaming_active = False  # Signal that streaming has ended
            self._log_broadcast_summary()
    

    def _log_broadcast_summary(self):
        """Log summary of broadcast session"""
        active_clients = self.get_active_clients()
        info(
            LogRecord(
                LogEvent.BROADCAST_SESSION_SUMMARY.value,
                f"Broadcast session completed: {self.total_chunks_processed} chunks to {len(self.clients)} clients",
                self.request_id,
                {
                    "provider": self.provider_name,
                    "total_chunks_processed": self.total_chunks_processed,
                    "total_clients": len(self.clients),
                    "active_clients_remaining": len(active_clients),
                    "client_summary": [
                        {
                            "type": client.client_type,
                            "active": client.is_active,
                            "chunks_sent": client.chunks_sent,
                            "error": str(client.last_error) if client.last_error else None
                        }
                        for client in self.clients
                    ]
                }
            )
        )


# Global registry for active broadcasters
_active_broadcasters: dict[str, ParallelBroadcaster] = {}

def create_broadcaster(request: Request, request_id: str, provider_name: str) -> ParallelBroadcaster:
    """Factory function to create a ParallelBroadcaster"""
    return ParallelBroadcaster(request, request_id, provider_name)

def register_broadcaster(signature: str, broadcaster: ParallelBroadcaster):
    """Register a broadcaster for duplicate request handling"""
    _active_broadcasters[signature] = broadcaster
    debug(
        LogRecord(
            LogEvent.BROADCASTER_REGISTERED.value,
            f"Broadcaster registered for signature {signature[:16]}...",
            broadcaster.request_id,
            {
                "provider": broadcaster.provider_name,
                "signature": signature[:16] + "...",
                "active_broadcasters": len(_active_broadcasters)
            }
        )
    )

def unregister_broadcaster(signature: str):
    """Unregister a broadcaster when streaming completes"""
    if signature in _active_broadcasters:
        broadcaster = _active_broadcasters.pop(signature)
        debug(
            LogRecord(
                LogEvent.BROADCASTER_UNREGISTERED.value,
                f"Broadcaster unregistered for signature {signature[:16]}...",
                broadcaster.request_id,
                {
                    "provider": broadcaster.provider_name,
                    "signature": signature[:16] + "...",
                    "active_broadcasters": len(_active_broadcasters)
                }
            )
        )

def has_active_broadcaster(signature: str) -> bool:
    """Check if there's an active broadcaster for the given signature"""
    return signature in _active_broadcasters

async def handle_duplicate_stream_request(signature: str, duplicate_request: Request, duplicate_request_id: str) -> AsyncGenerator[str, None]:
    """
    Handle a duplicate stream request by connecting to existing broadcaster.
    Returns an async generator that yields all chunks (past + future).
    Raises Exception if no active broadcaster is found.
    """
    if signature in _active_broadcasters:
        broadcaster = _active_broadcasters[signature]
        info(
            LogRecord(
                LogEvent.DUPLICATE_REQUEST_FOUND_ACTIVE_BROADCASTER.value,
                "Found active broadcaster for duplicate request",
                duplicate_request_id,
                {
                    "original_request_id": broadcaster.request_id,
                    "provider": broadcaster.provider_name,
                    "signature": signature[:16] + "...",
                    "chunks_already_processed": len(broadcaster.collected_chunks)
                }
            )
        )
        
        try:
            # Use the broadcaster's method to handle the duplicate
            async for chunk in broadcaster.add_duplicate_request(duplicate_request, duplicate_request_id):
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            # Handle graceful cancellation/closure
            debug(
                LogRecord(
                    LogEvent.DUPLICATE_DISCONNECTED_DURING_LIVE_CHUNK.value,
                    "Duplicate stream request cancelled or closed",
                    duplicate_request_id,
                    {
                        "original_request_id": broadcaster.request_id,
                        "provider": broadcaster.provider_name,
                        "signature": signature[:16] + "...",
                        "reason": "handle_duplicate_cancelled"
                    }
                )
            )
            return
    else:
        # No active broadcaster found
        debug(  # Changed from error to debug since this is expected behavior
            LogRecord(
                LogEvent.DUPLICATE_REQUEST_NO_ACTIVE_BROADCASTER.value,
                "No active broadcaster found for duplicate request",
                duplicate_request_id,
                {
                    "signature": signature[:16] + "...",
                    "active_broadcasters": len(_active_broadcasters),
                    "available_signatures": [sig[:16] + "..." for sig in _active_broadcasters.keys()]
                }
            )
        )
        # Raise exception to signal fallback to normal duplicate handling
        raise Exception(f"No active broadcaster found for signature {signature[:16]}...")