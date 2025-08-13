"""
OAuth Manager for Claude Code Provider Balancer
Handles OAuth 2.0 authentication flow for Claude Code Official provider.
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

import httpx

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    # Import will be available after this block

from utils import info, warning, error, debug, LogRecord, LogEvent

# Warn about missing keyring if needed
if not KEYRING_AVAILABLE:
    warning(LogRecord(
        event=LogEvent.OAUTH_KEYRING_UNAVAILABLE.value,
        message="keyring library not available. Token persistence will be disabled."
    ))

# OAuth constants from claude-code-login
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"

@dataclass
class TokenCredentials:
    """OAuth token credentials"""
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp
    scopes: List[str]
    account_id: Optional[str] = None
    account_email: Optional[str] = None
    account_name: Optional[str] = None
    # Usage statistics
    usage_count: int = 0
    last_used: Optional[int] = None  # Unix timestamp
    created_at: Optional[int] = None  # Unix timestamp
    
    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if token is expired or will expire within buffer_seconds"""
        return time.time() + buffer_seconds >= self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token, 
            "expires_at": self.expires_at,
            "scopes": self.scopes,
            "account_id": self.account_id,
            "account_email": self.account_email,
            "account_name": self.account_name,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenCredentials':
        """Create from dictionary"""
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            scopes=data["scopes"],
            account_id=data.get("account_id"),
            account_email=data.get("account_email"),
            account_name=data.get("account_name"),
            usage_count=data.get("usage_count", 0),
            last_used=data.get("last_used"),
            created_at=data.get("created_at")
        )

@dataclass
class OAuthState:
    """OAuth state for PKCE flow"""
    state: str
    code_verifier: str
    code_challenge: str
    expires_at: int

class OAuthManager:
    """Manages OAuth 2.0 authentication for Claude Code Official"""
    
    def __init__(self, enable_persistence: bool = True, proxy: Optional[str] = None):
        self.token_credentials: List[TokenCredentials] = []
        self.current_token_index = 0
        self.oauth_state: Optional[OAuthState] = None
        self._lock = threading.Lock()
        self._refresh_tasks: Dict[int, asyncio.Task] = {}
        
        # Keyring settings
        self.enable_persistence = enable_persistence and KEYRING_AVAILABLE
        self.service_name = "claude-code-balancer"
        
        # Proxy settings
        self.proxy = proxy
        
        # Load tokens from keyring on startup
        if self.enable_persistence:
            self._load_from_keyring()
    
    def _save_to_keyring(self):
        """Save all tokens to system keyring"""
        if not self.enable_persistence:
            return
        
        try:
            with self._lock:
                # Save tokens as JSON
                tokens_data = [creds.to_dict() for creds in self.token_credentials]
                metadata = {
                    "current_token_index": self.current_token_index,
                    "last_saved": int(time.time())
                }
                
                keyring_data = {
                    "tokens": tokens_data,
                    "metadata": metadata
                }
                
                keyring.set_password(
                    self.service_name,
                    "oauth_tokens", 
                    json.dumps(keyring_data)
                )
                
                debug(LogRecord(
                    event=LogEvent.OAUTH_TOKENS_SAVED.value,
                    message=f"Saved {len(tokens_data)} tokens to keyring"
                ))
                
        except Exception as e:
            warning(LogRecord(
                event=LogEvent.OAUTH_SAVE_FAILED.value,
                message=f"Failed to save tokens to keyring: {str(e)}"
            ))
    
    async def _safe_save_to_keyring(self):
        """Safely save to keyring without holding lock for extended periods"""
        if not self.enable_persistence:
            return
        
        try:
            # Get a snapshot of data with minimal lock time
            tokens_data = None
            metadata = None
            
            with self._lock:
                tokens_data = [creds.to_dict() for creds in self.token_credentials]
                metadata = {
                    "current_token_index": self.current_token_index,
                    "last_saved": int(time.time())
                }
            
            # Perform keyring operation outside of lock
            keyring_data = {
                "tokens": tokens_data,
                "metadata": metadata
            }
            
            keyring.set_password(
                self.service_name,
                "oauth_tokens", 
                json.dumps(keyring_data)
            )
            
            debug(LogRecord(
                event=LogEvent.OAUTH_TOKENS_SAVED_ASYNC.value,
                message=f"Saved {len(tokens_data)} tokens usage statistics to keyring (async)"
            ))
            
        except Exception as e:
            warning(LogRecord(
                event=LogEvent.OAUTH_SAVE_FAILED_ASYNC.value,
                message=f"Failed to save tokens usage statistics to keyring (async): {str(e)}"
            ))
    
    def _load_from_keyring(self):
        """Load tokens from system keyring"""
        if not self.enable_persistence:
            return
        
        try:
            stored_data = keyring.get_password(self.service_name, "oauth_tokens")
            if not stored_data:
                debug(LogRecord(
                    event=LogEvent.OAUTH_NO_TOKENS_FOUND.value,
                    message="No tokens found in keyring"
                ))
                return
            
            keyring_data = json.loads(stored_data)
            tokens_data = keyring_data.get("tokens", [])
            metadata = keyring_data.get("metadata", {})
            
            with self._lock:
                # Load tokens
                self.token_credentials = [
                    TokenCredentials.from_dict(token_data) 
                    for token_data in tokens_data
                ]
                
                # Restore current index
                self.current_token_index = metadata.get("current_token_index", 0)
                
                # Validate index
                if self.current_token_index >= len(self.token_credentials):
                    self.current_token_index = 0
            
            last_saved = metadata.get("last_saved", 0)
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_saved))
            
            info(LogRecord(
                event=LogEvent.OAUTH_TOKENS_LOADED.value,
                message=f"Loaded {len(self.token_credentials)} tokens from keyring (saved: {saved_time})"
            ))
            
            # Clean up expired tokens
            self._cleanup_expired_tokens()
                
        except Exception as e:
            warning(LogRecord(
                event=LogEvent.OAUTH_LOAD_FAILED.value,
                message=f"Failed to load tokens from keyring: {str(e)}"
            ))
    
    def _cleanup_expired_tokens(self):
        """Remove completely expired tokens (no refresh possible)"""
        with self._lock:
            # Keep tokens that are not expired or can be refreshed
            valid_tokens = []
            removed_count = 0
            
            for creds in self.token_credentials:
                # Keep token if not expired (with generous buffer) or has refresh token
                if not creds.is_expired(3600) or creds.refresh_token:
                    valid_tokens.append(creds)
                else:
                    removed_count += 1
            
            if removed_count > 0:
                self.token_credentials = valid_tokens
                
                # Adjust current index
                if self.current_token_index >= len(self.token_credentials):
                    self.current_token_index = 0
                
                info(LogRecord(
                    event=LogEvent.OAUTH_TOKENS_CLEANED.value,
                    message=f"Cleaned up {removed_count} expired tokens from storage"
                ))
        
        # Save updated list back to keyring OUTSIDE of lock
        if removed_count > 0 and self.enable_persistence:
            # Use the synchronous save method but schedule it to avoid deadlock
            try:
                # Get snapshot of current data without holding lock for keyring operation
                with self._lock:
                    tokens_data = [creds.to_dict() for creds in self.token_credentials]
                    metadata = {
                        "current_token_index": self.current_token_index,
                        "last_saved": int(time.time())
                    }
                
                # Perform keyring operation outside of lock
                keyring_data = {
                    "tokens": tokens_data,
                    "metadata": metadata
                }
                
                keyring.set_password(
                    self.service_name,
                    "oauth_tokens", 
                    json.dumps(keyring_data)
                )
                
                debug(LogRecord(
                    event=LogEvent.OAUTH_TOKENS_SAVED_AFTER_CLEANUP.value,
                    message=f"Saved {len(tokens_data)} tokens to keyring after cleanup"
                ))
                
            except Exception as e:
                warning(LogRecord(
                    event=LogEvent.OAUTH_SAVE_FAILED_AFTER_CLEANUP.value,
                    message=f"Failed to save tokens to keyring after cleanup: {str(e)}"
                ))
        
    def generate_pkce_challenge(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        # Generate code verifier (43-128 characters, base64url encoded)
        code_verifier = secrets.token_urlsafe(32)
        
        # Create code challenge (SHA256 of verifier, base64url encoded)
        import base64
        challenge = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(challenge).decode('ascii').rstrip("=")
        
        return code_verifier, code_challenge
    
    def generate_login_url(self) -> str:
        """Generate OAuth authorization URL for Claude Code authentication"""
        # Generate secure random values
        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = self.generate_pkce_challenge()
        
        # Store state for verification
        self.oauth_state = OAuthState(
            state=state,
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            expires_at=int(time.time()) + 600  # 10 minutes
        )
        
        # Build OAuth URL
        params = {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state
        }
        
        url = f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"
        
        info(LogRecord(
            event=LogEvent.OAUTH_URL_GENERATED.value,
            message="Generated OAuth login URL for Claude Code authentication"
        ))
        debug(LogRecord(
            event=LogEvent.OAUTH_URL_DEBUG.value,
            message=f"OAuth URL: {url}"
        ))
        
        return url
    
    async def exchange_code(self, authorization_code: str, account_email: str) -> Optional[TokenCredentials]:
        """Exchange authorization code for access and refresh tokens"""
        if not self.oauth_state:
            error(LogRecord(
                event=LogEvent.OAUTH_NO_STATE.value,
                message="No OAuth state found. Please generate login URL first."
            ))
            return None
        
        # Check if state is expired
        if time.time() > self.oauth_state.expires_at:
            error(LogRecord(
                event=LogEvent.OAUTH_STATE_EXPIRED.value,
                message="OAuth state has expired (older than 10 minutes). Please generate new login URL."
            ))
            return None
        
        # Clean authorization code
        cleaned_code = authorization_code.split('#')[0].split('&')[0]
        
        params = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": cleaned_code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": self.oauth_state.code_verifier,
            "state": self.oauth_state.state
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://claude.ai/",
            "Origin": "https://claude.ai"
        }
        
        try:
            client_kwargs = {}
            if self.proxy:
                client_kwargs["proxy"] = self.proxy
            
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    OAUTH_TOKEN_URL,
                    headers=headers,
                    json=params,
                    timeout=30
                )
                
                if not response.is_success:
                    error_text = response.text
                    error(LogRecord(
                        event=LogEvent.OAUTH_TOKEN_EXCHANGE_FAILED.value,
                        message=f"Token exchange failed: {response.status_code} - {error_text}"
                    ))
                    return None
                
                token_data = response.json()
                
                # Use the required account email directly as account identifier
                final_user_email = account_email
                account_id = account_email
                
                # Check for existing account and remove old tokens
                # First, find tasks to cancel WITHOUT holding the lock
                tasks_to_cancel = []
                existing_indices = []
                with self._lock:
                    # Find existing tokens for this account using multiple criteria
                    for i, existing_creds in enumerate(self.token_credentials):
                        is_duplicate = False
                        
                        # Method 1: Same email (most reliable) - use manual email with higher priority
                        if (existing_creds.account_email and final_user_email and 
                            existing_creds.account_email.lower() == final_user_email.lower()):
                            is_duplicate = True
                            info(LogRecord(
                                event=LogEvent.OAUTH_DUPLICATE_FOUND_EMAIL.value,
                                message=f"Found duplicate by email: {existing_creds.account_email}"
                            ))
                        
                        # Method 2: Same access token prefix (same user, same session)
                        elif (existing_creds.access_token and 
                              existing_creds.access_token[:20] == token_data["access_token"][:20]):
                            is_duplicate = True
                            info(LogRecord(
                                event=LogEvent.OAUTH_DUPLICATE_FOUND_TOKEN.value,
                                message=f"Found duplicate by token prefix: {existing_creds.access_token[:10]}..."
                            ))
                        
                        # Method 3: Same refresh token (definitely same user)
                        elif (existing_creds.refresh_token and 
                              existing_creds.refresh_token == token_data["refresh_token"]):
                            is_duplicate = True
                            info(LogRecord(
                                event=LogEvent.OAUTH_DUPLICATE_FOUND_REFRESH.value,
                                message="Found duplicate by refresh token"
                            ))
                        
                        # Method 4: Same token fingerprint (based on refresh token)
                        elif (hasattr(existing_creds, 'token_fingerprint') and existing_creds.token_fingerprint and
                              token_data.get("refresh_token")):
                            # Create fingerprint for new token
                            import hashlib
                            new_token_hash = hashlib.sha256(token_data["refresh_token"].encode()).hexdigest()
                            new_fingerprint = new_token_hash[:16]
                            if existing_creds.token_fingerprint == new_fingerprint:
                                is_duplicate = True
                                info(LogRecord(
                                    event=LogEvent.OAUTH_DUPLICATE_FOUND_FINGERPRINT.value,
                                    message=f"Found duplicate by token fingerprint: {new_fingerprint}"
                                ))
                        
                        # Method 5: Same account_id (fallback)
                        elif existing_creds.account_id == account_id:
                            is_duplicate = True
                            info(LogRecord(
                                event=LogEvent.OAUTH_DUPLICATE_FOUND_ACCOUNT.value,
                                message=f"Found duplicate by account_id: {account_id}"
                            ))
                        
                        if is_duplicate:
                            existing_indices.append(i)
                            # Collect tasks to cancel
                            if i in self._refresh_tasks:
                                tasks_to_cancel.append(self._refresh_tasks[i])
                                del self._refresh_tasks[i]
                
                # Cancel tasks outside of lock to avoid deadlock
                for task in tasks_to_cancel:
                    task.cancel()
                
                # Now remove tokens while holding the lock
                with self._lock:
                    # Remove existing tokens for this account (from highest index to lowest)
                    for i in sorted(existing_indices, reverse=True):
                        old_creds = self.token_credentials[i]
                        info(LogRecord(
                            event=LogEvent.OAUTH_TOKEN_REMOVED.value,
                            message=f"Removing existing token for account {old_creds.account_id}"
                        ))
                        
                        # Remove from list
                        self.token_credentials.pop(i)
                        
                        # Adjust current index if needed
                        if self.current_token_index >= len(self.token_credentials):
                            self.current_token_index = 0
                
                # Create new credentials with current timestamp
                current_time = int(time.time())
                credentials = TokenCredentials(
                    access_token=token_data["access_token"],
                    refresh_token=token_data["refresh_token"],
                    expires_at=current_time + token_data["expires_in"],
                    scopes=token_data.get("scope", SCOPES).split(" "),
                    account_id=account_id,
                    account_email=final_user_email,
                    account_name=None,  # No longer fetched from API
                    usage_count=0,
                    last_used=None,
                    created_at=current_time
                )
                
                # Create a token fingerprint for duplicate detection based on refresh token
                # This should be consistent for the same user across multiple authorizations
                import hashlib
                if token_data.get("refresh_token"):
                    token_hash = hashlib.sha256(token_data["refresh_token"].encode()).hexdigest()
                    credentials.token_fingerprint = token_hash[:16]  # First 16 chars for identification
                else:
                    credentials.token_fingerprint = None
                
                # Add to memory storage
                with self._lock:
                    self.token_credentials.append(credentials)
                    info(LogRecord(
                        event=LogEvent.OAUTH_TOKEN_ADDED.value,
                        message=f"Added new token for account {credentials.account_id}"
                    ))
                
                # Save to keyring OUTSIDE of lock
                if self.enable_persistence:
                    try:
                        # Get snapshot of current data without holding lock for keyring operation
                        with self._lock:
                            tokens_data = [creds.to_dict() for creds in self.token_credentials]
                            metadata = {
                                "current_token_index": self.current_token_index,
                                "last_saved": int(time.time())
                            }
                        
                        # Perform keyring operation outside of lock
                        keyring_data = {
                            "tokens": tokens_data,
                            "metadata": metadata
                        }
                        
                        keyring.set_password(
                            self.service_name,
                            "oauth_tokens", 
                            json.dumps(keyring_data)
                        )
                        
                        debug(LogRecord(
                            event=LogEvent.OAUTH_TOKENS_SAVED_AFTER_EXCHANGE.value,
                            message=f"Saved {len(tokens_data)} tokens to keyring after exchange"
                        ))
                        
                    except Exception as e:
                        warning(LogRecord(
                            event=LogEvent.OAUTH_SAVE_FAILED_AFTER_EXCHANGE.value,
                            message=f"Failed to save tokens to keyring after exchange: {str(e)}"
                        ))
                
                # Clear OAuth state
                self.oauth_state = None
                
                info(LogRecord(
                    event=LogEvent.OAUTH_EXCHANGE_SUCCESS.value,
                    message=f"Successfully exchanged authorization code for tokens. Account ID: {credentials.account_id}"
                ))
                info(LogRecord(
                    event=LogEvent.OAUTH_TOKEN_EXPIRY.value,
                    message=f"Token expires at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(credentials.expires_at))}"
                ))
                
                # Start auto-refresh for the new token
                credentials_index = len(self.token_credentials) - 1  # The newly added token is at the end
                if credentials_index not in self._refresh_tasks:
                    task = asyncio.create_task(self._auto_refresh_loop(credentials))
                    self._refresh_tasks[credentials_index] = task
                    debug(LogRecord(
                        event=LogEvent.OAUTH_AUTO_REFRESH_STARTED.value,
                        message=f"Started auto-refresh for newly acquired token: {credentials.account_id}"
                    ))
                
                return credentials
                
        except Exception as e:
            error(LogRecord(
                event=LogEvent.OAUTH_EXCHANGE_ERROR.value,
                message=f"Error exchanging authorization code: {str(e)}"
            ))
            return None
    
    async def refresh_token(self, credentials: TokenCredentials) -> tuple[Optional[TokenCredentials], Optional[str]]:
        """Refresh access token using refresh token
        
        Returns:
            tuple[Optional[TokenCredentials], Optional[str]]: (credentials, error_message)
        """
        params = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": credentials.refresh_token
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*"
        }
        
        try:
            client_kwargs = {}
            if self.proxy:
                client_kwargs["proxy"] = self.proxy
            
            debug(LogRecord(
                event=LogEvent.OAUTH_REFRESH_REQUEST.value,
                message=f"Token refresh request for {credentials.account_id} with proxy: {self.proxy}"
            ))
            
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    OAUTH_TOKEN_URL,
                    headers=headers,
                    json=params,
                    timeout=30
                )
                
                debug(LogRecord(
                    event=LogEvent.OAUTH_REFRESH_RESPONSE_STATUS.value,
                    message=f"Token refresh response status: {response.status_code}"
                ))
                # Redact sensitive token information from response
                response_text = response.text
                try:
                    # Try to parse as JSON and redact tokens
                    import re
                    # Redact access_token and refresh_token fields
                    response_text = re.sub(r'"access_token"\s*:\s*"[^"]*"', '"access_token":"[REDACTED]"', response_text)
                    response_text = re.sub(r'"refresh_token"\s*:\s*"[^"]*"', '"refresh_token":"[REDACTED]"', response_text)
                except:
                    # If regex fails, truncate the response
                    response_text = response_text[:200] + "..." if len(response_text) > 200 else response_text
                
                debug(LogRecord(
                    event=LogEvent.OAUTH_REFRESH_RESPONSE_TEXT.value,
                    message=f"Token refresh response text: {response_text}"
                ))
                
                if not response.is_success:
                    error_text = response.text
                    error_message = f"HTTP {response.status_code}: {error_text}"
                    error(LogRecord(
                        event=LogEvent.OAUTH_REFRESH_FAILED.value,
                        message=f"Token refresh failed for {credentials.account_id}: {error_message}"
                    ))
                    return None, error_message
                
                token_data = response.json()
                debug(LogRecord(
                    event=LogEvent.OAUTH_REFRESH_SUCCESSFUL.value,
                    message="Token refresh successful, received new tokens"
                ))
                
                # Update credentials
                credentials.access_token = token_data["access_token"]
                if "refresh_token" in token_data:
                    credentials.refresh_token = token_data["refresh_token"]
                credentials.expires_at = int(time.time()) + token_data["expires_in"]
                
                # Schedule keyring save without holding lock to avoid deadlock
                if self.enable_persistence:
                    # Create a background task to save to keyring to avoid blocking
                    asyncio.create_task(self._safe_save_to_keyring())
                
                info(LogRecord(
                    event=LogEvent.OAUTH_TOKEN_REFRESHED.value,
                    message=f"Successfully refreshed token for {credentials.account_id}"
                ))
                debug(LogRecord(
                    event=LogEvent.OAUTH_NEW_TOKEN_EXPIRY.value,
                    message=f"New token expires at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(credentials.expires_at))}"
                ))
                
                return credentials, None
                
        except Exception as e:
            error_message = f"Network/Connection error: {str(e)}"
            error(LogRecord(
                event=LogEvent.OAUTH_REFRESH_ERROR.value,
                message=f"Error refreshing token for {credentials.account_id}: {error_message}"
            ))
            return None, error_message
    
    def get_current_token(self) -> Optional[str]:
        """Get current access token using round-robin strategy"""
        with self._lock:
            if not self.token_credentials:
                return None
            
            # Find next healthy token
            start_index = self.current_token_index
            attempts = 0
            
            while attempts < len(self.token_credentials):
                credentials = self.token_credentials[self.current_token_index]
                
                # Check if token is not expired (with 5-minute buffer)
                if not credentials.is_expired(300):
                    # Update usage statistics
                    current_time = int(time.time())
                    credentials.usage_count += 1
                    credentials.last_used = current_time
                    
                    # Move to next token for next request
                    self.current_token_index = (self.current_token_index + 1) % len(self.token_credentials)
                    debug(LogRecord(
                        event=LogEvent.OAUTH_TOKEN_USED.value,
                        message=f"Using token from {credentials.account_id} (usage: {credentials.usage_count})"
                    ))
                    
                    # Save updated statistics to keyring (async to avoid blocking)
                    if self.enable_persistence:
                        import asyncio
                        try:
                            # Schedule async save without blocking
                            loop = asyncio.get_event_loop()
                            asyncio.create_task(self._safe_save_to_keyring())
                        except Exception:
                            # If async fails, skip saving to avoid blocking
                            pass
                    
                    return credentials.access_token
                
                # Token is expired, try next one
                self.current_token_index = (self.current_token_index + 1) % len(self.token_credentials)
                attempts += 1
            
            warning(LogRecord(
                event=LogEvent.OAUTH_ALL_TOKENS_INVALID.value,
                message="All tokens are expired or invalid"
            ))
            return None
    
    def get_token_by_email(self, account_email: str) -> Optional[str]:
        """根据账户邮箱获取对应的access token"""
        if not account_email:
            debug(LogRecord(
                event=LogEvent.OAUTH_GET_TOKEN_BY_EMAIL_NO_EMAIL.value,
                message="No account_email provided, falling back to round-robin"
            ))
            return self.get_current_token()
        
        with self._lock:
            if not self.token_credentials:
                debug(LogRecord(
                    event=LogEvent.OAUTH_GET_TOKEN_BY_EMAIL_NO_TOKENS.value,
                    message=f"No OAuth tokens available for account {account_email}"
                ))
                return None
            
            # Find token for specific account
            for creds in self.token_credentials:
                if (creds.account_email and creds.account_email.lower() == account_email.lower()) or \
                   (creds.account_id and creds.account_id.lower() == account_email.lower()):
                    
                    # Check if token is not expired (with 5-minute buffer)
                    if not creds.is_expired(300):
                        # Update usage statistics
                        current_time = int(time.time())
                        creds.usage_count += 1
                        creds.last_used = current_time
                        
                        debug(LogRecord(
                            event=LogEvent.OAUTH_TOKEN_USED_BY_EMAIL.value,
                            message=f"Using token from {creds.account_email or creds.account_id} (usage: {creds.usage_count})"
                        ))
                        
                        # Save updated statistics to keyring (async to avoid blocking)
                        if self.enable_persistence:
                            import asyncio
                            try:
                                # Schedule async save without blocking
                                loop = asyncio.get_event_loop()
                                asyncio.create_task(self._safe_save_to_keyring())
                            except Exception:
                                # If async fails, skip saving to avoid blocking
                                pass
                        
                        return creds.access_token
                    else:
                        warning(LogRecord(
                            event=LogEvent.OAUTH_TOKEN_EXPIRED_BY_EMAIL.value,
                            message=f"Token for account {account_email} is expired"
                        ))
                        return None
            
            # Account not found
            warning(LogRecord(
                event=LogEvent.OAUTH_ACCOUNT_NOT_FOUND.value,
                message=f"No OAuth token found for account {account_email}"
            ))
            return None
    
    def get_tokens_status(self) -> List[Dict[str, Any]]:
        """Get status of all stored tokens (safe, no sensitive info)"""
        with self._lock:
            status = []
            current_time = time.time()
            
            for i, creds in enumerate(self.token_credentials):
                # Calculate time information
                expires_in_seconds = max(0, creds.expires_at - current_time)
                # Use actual created_at if available, otherwise estimate from expires_at
                actual_created_time = creds.created_at or (creds.expires_at - 3600)  # Assuming 1 hour token lifespan
                
                token_status = {
                    "account_email": creds.account_email or creds.account_id,  # Use email as primary identifier
                    "index": i,
                    "is_current": i == self.current_token_index,
                    
                    # Time information
                    "created_at": int(actual_created_time),
                    "expires_at": creds.expires_at,
                    "expires_in_seconds": int(expires_in_seconds),
                    "expires_in_minutes": round(expires_in_seconds / 60, 1),
                    "expires_in_human": self._format_duration(expires_in_seconds),
                    
                    # Status
                    "is_expired": creds.is_expired(),
                    "is_healthy": not creds.is_expired(300),  # 5 minute buffer
                    "will_expire_soon": creds.is_expired(900),  # 15 minute warning
                    
                    # Safe token info (masked)
                    "access_token_preview": f"{creds.access_token[:8]}...{creds.access_token[-4:]}" if creds.access_token else None,
                    "refresh_token_preview": f"{creds.refresh_token[:8]}...{creds.refresh_token[-4:]}" if creds.refresh_token else None,
                    
                    # Permissions
                    "scopes": creds.scopes,
                    "scope_count": len(creds.scopes),
                    
                    # Usage statistics (now tracked)
                    "usage_count": creds.usage_count,
                    "last_used": self._format_last_used(creds.last_used) if creds.last_used else "Never",
                    "last_used_timestamp": creds.last_used
                }
                
                # Add account name if available (but no longer primary identifier)
                if creds.account_name:
                    token_status["account_name"] = creds.account_name
                
                status.append(token_status)
            
            return status
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds <= 0:
            return "已过期"
        elif seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds/60)}分钟"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分钟"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}天{hours}小时"
    
    def _format_last_used(self, last_used_timestamp: int) -> str:
        """Format last used time in human readable format"""
        current_time = int(time.time())
        time_diff = current_time - last_used_timestamp
        
        if time_diff <= 0:
            return "刚刚"
        elif time_diff < 60:
            return f"{time_diff}秒前"
        elif time_diff < 3600:
            minutes = int(time_diff / 60)
            seconds = time_diff % 60
            if minutes == 0:
                return f"{seconds}秒前"
            elif seconds == 0:
                return f"{minutes}分钟前"
            else:
                return f"{minutes}分{seconds}秒前"
        elif time_diff < 86400:
            hours = int(time_diff / 3600)
            return f"{hours}小时前"
        else:
            days = int(time_diff / 86400)
            if days == 1:
                return "昨天"
            elif days < 7:
                return f"{days}天前"
            else:
                # For older usage, show actual date
                import datetime
                dt = datetime.datetime.fromtimestamp(last_used_timestamp)
                return dt.strftime("%Y-%m-%d %H:%M")
    
    async def start_auto_refresh(self):
        """Start automatic token refresh for all credentials"""
        for i, credentials in enumerate(self.token_credentials):
            if i not in self._refresh_tasks:
                task = asyncio.create_task(self._auto_refresh_loop(credentials))
                self._refresh_tasks[i] = task
                debug(LogRecord(
                    event=LogEvent.OAUTH_AUTO_REFRESH_STARTED.value,
                    message=f"Started auto-refresh for {credentials.account_id}"
                ))
    
    async def _auto_refresh_loop(self, credentials: TokenCredentials):
        """Auto-refresh loop for a single token"""
        while True:
            try:
                # Check if token needs refresh (5 minutes before expiry)
                if credentials.is_expired(300):
                    info(LogRecord(
                        event=LogEvent.OAUTH_TOKEN_NEEDS_REFRESH.value,
                        message=f"Token for {credentials.account_id} needs refresh"
                    ))
                    
                    refreshed, error_details = await self.refresh_token(credentials)
                    if not refreshed:
                        error(LogRecord(
                            event=LogEvent.OAUTH_AUTO_REFRESH_FAILED.value,
                            message=f"Failed to refresh token for {credentials.account_id}"
                        ))
                        # Wait 1 hour before retry
                        await asyncio.sleep(3600)
                        continue
                
                # Calculate next refresh time (5 minutes before expiry)
                sleep_time = max(60, credentials.expires_at - time.time() - 300)
                debug(LogRecord(
                    event=LogEvent.OAUTH_NEXT_REFRESH_SCHEDULED.value,
                    message=f"Next refresh for {credentials.account_id} in {sleep_time/60:.1f} minutes"
                ))
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                info(LogRecord(
                    event=LogEvent.OAUTH_AUTO_REFRESH_CANCELLED.value,
                    message=f"Auto-refresh cancelled for {credentials.account_id} (likely due to manual token refresh or shutdown)"
                ))
                break
            except Exception as e:
                error(LogRecord(
                    event=LogEvent.OAUTH_AUTO_REFRESH_LOOP_ERROR.value,
                    message=f"Error in auto-refresh loop for {credentials.account_id}: {str(e)}"
                ))
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def refresh_token_by_email(self, account_email: str) -> tuple[Optional[TokenCredentials], Optional[str]]:
        """Refresh token for a specific account by email
        
        Returns:
            tuple[Optional[TokenCredentials], Optional[str]]: (credentials, error_message)
        """
        target_credentials = None
        
        # Find the credentials by account email
        with self._lock:
            for creds in self.token_credentials:
                if (creds.account_email and creds.account_email.lower() == account_email.lower()) or \
                   (creds.account_id and creds.account_id.lower() == account_email.lower()):
                    target_credentials = creds
                    break
        
        if not target_credentials:
            error_msg = f"No token found for account email: {account_email}"
            warning(LogRecord(
                event=LogEvent.OAUTH_REFRESH_BY_EMAIL_NOT_FOUND.value,
                message=error_msg
            ))
            return None, error_msg
        
        if not target_credentials.refresh_token:
            error_msg = f"No refresh token available for account: {account_email}"
            warning(LogRecord(
                event=LogEvent.OAUTH_REFRESH_BY_EMAIL_NO_REFRESH_TOKEN.value,
                message=error_msg
            ))
            return None, error_msg
        
        info(LogRecord(
            event=LogEvent.OAUTH_REFRESH_BY_EMAIL_START.value,
            message=f"Starting manual token refresh for account: {account_email}"
        ))
        
        # Refresh the token
        refreshed_credentials, error_details = await self.refresh_token(target_credentials)
        
        if refreshed_credentials:
            info(LogRecord(
                event=LogEvent.OAUTH_REFRESH_BY_EMAIL_SUCCESS.value,
                message=f"Successfully refreshed token for account: {account_email}"
            ))
            return refreshed_credentials, None
        else:
            error(LogRecord(
                event=LogEvent.OAUTH_REFRESH_BY_EMAIL_FAILED.value,
                message=f"Failed to refresh token for account: {account_email}"
            ))
            return None, error_details

    def remove_token(self, account_email: str) -> bool:
        """Remove token by account email"""
        # First, find the task to cancel WITHOUT holding the lock
        task_to_cancel = None
        with self._lock:
            for i, creds in enumerate(self.token_credentials):
                if creds.account_id == account_email:
                    # Get task to cancel later
                    if i in self._refresh_tasks:
                        task_to_cancel = self._refresh_tasks[i]
                        del self._refresh_tasks[i]
                    break
            else:
                # Account not found
                return False
        
        # Cancel task outside of lock to avoid deadlock
        if task_to_cancel:
            task_to_cancel.cancel()
        
        # Now remove data while holding the lock
        with self._lock:
            for i, creds in enumerate(self.token_credentials):
                if creds.account_id == account_email:
                    # Remove from list
                    self.token_credentials.pop(i)
                    
                    # Adjust current index if needed
                    if self.current_token_index >= len(self.token_credentials):
                        self.current_token_index = 0
                    
                    info(LogRecord(
                        event=LogEvent.OAUTH_TOKEN_REMOVED_BY_EMAIL.value,
                        message=f"Removed token for {account_email}"
                    ))
                    
                    # Save to keyring after removal OUTSIDE of lock
                    if self.enable_persistence:
                        try:
                            # Get snapshot of current data without holding lock for keyring operation
                            with self._lock:
                                tokens_data = [creds.to_dict() for creds in self.token_credentials]
                                metadata = {
                                    "current_token_index": self.current_token_index,
                                    "last_saved": int(time.time())
                                }
                            
                            # Perform keyring operation outside of lock
                            keyring_data = {
                                "tokens": tokens_data,
                                "metadata": metadata
                            }
                            
                            keyring.set_password(
                                self.service_name,
                                "oauth_tokens", 
                                json.dumps(keyring_data)
                            )
                            
                            debug(LogRecord(
                                event=LogEvent.OAUTH_TOKENS_SAVED_AFTER_REMOVAL.value,
                                message=f"Saved {len(tokens_data)} tokens to keyring after removal"
                            ))
                            
                        except Exception as e:
                            warning(LogRecord(
                                event=LogEvent.OAUTH_SAVE_FAILED_AFTER_REMOVAL.value,
                                message=f"Failed to save tokens to keyring after removal: {str(e)}"
                            ))
                    
                    return True
            
            return False
    
    def clear_all_tokens(self):
        """Clear all stored tokens"""
        # First, cancel all refresh tasks WITHOUT holding the lock
        tasks_to_cancel = []
        with self._lock:
            tasks_to_cancel = list(self._refresh_tasks.values())
            self._refresh_tasks.clear()
        
        # Cancel tasks outside of lock to avoid deadlock
        for task in tasks_to_cancel:
            task.cancel()
        
        # Now clear data while holding the lock
        with self._lock:
            # Clear credentials
            self.token_credentials.clear()
            self.current_token_index = 0
            
            info(LogRecord(
                event=LogEvent.OAUTH_ALL_TOKENS_CLEARED.value,
                message="Cleared all stored tokens"
            ))
        
        # Clear from keyring OUTSIDE of lock
        if self.enable_persistence:
            try:
                # Get snapshot of current data without holding lock for keyring operation
                with self._lock:
                    tokens_data = [creds.to_dict() for creds in self.token_credentials]
                    metadata = {
                        "current_token_index": self.current_token_index,
                        "last_saved": int(time.time())
                    }
                
                # Perform keyring operation outside of lock
                keyring_data = {
                    "tokens": tokens_data,
                    "metadata": metadata
                }
                
                keyring.set_password(
                    self.service_name,
                    "oauth_tokens", 
                    json.dumps(keyring_data)
                )
                
                debug(LogRecord(
                    event=LogEvent.OAUTH_TOKENS_SAVED_AFTER_CLEAR.value,
                    message=f"Saved {len(tokens_data)} tokens to keyring after clear"
                ))
                
            except Exception as e:
                warning(LogRecord(
                    event=LogEvent.OAUTH_SAVE_FAILED_AFTER_CLEAR.value,
                    message=f"Failed to save tokens to keyring after clear: {str(e)}"
                ))

# Global OAuth manager instance will be created later with config
oauth_manager = None

def init_oauth_manager(config_settings: Optional[Dict[str, Any]] = None):
    """Initialize OAuth manager with configuration settings"""
    global oauth_manager
    
    # Don't reinitialize if already exists and has tokens
    if oauth_manager and oauth_manager.token_credentials:
        debug(LogRecord(
            event=LogEvent.OAUTH_EXISTING_TOKENS_FOUND.value,
            message="OAuth manager already initialized with tokens, skipping reinitialization"
        ))
        return oauth_manager
    
    # Default settings
    enable_persistence = True
    service_name = "claude-code-balancer"
    proxy = None
    
    # Load from config if provided
    if config_settings:
        oauth_config = config_settings.get('oauth', {})
        enable_persistence = oauth_config.get('enable_persistence', True)
        service_name = oauth_config.get('service_name', 'claude-code-balancer')
        proxy = oauth_config.get('proxy')
    
    oauth_manager = OAuthManager(enable_persistence=enable_persistence, proxy=proxy)
    oauth_manager.service_name = service_name
    
    debug(LogRecord(
        event=LogEvent.OAUTH_MANAGER_INITIALIZED.value,
        message=f"OAuth manager initialized (persistence: {enable_persistence}, service: {service_name}, proxy: {proxy})"
    ))
    
    return oauth_manager

def get_oauth_manager():
    """Get the current OAuth manager instance"""
    global oauth_manager
    return oauth_manager

async def start_oauth_auto_refresh(auto_refresh_enabled: bool = True):
    """Start auto-refresh for loaded tokens (should be called after app startup)"""
    global oauth_manager
    if oauth_manager and oauth_manager.token_credentials and auto_refresh_enabled:
        await oauth_manager.start_auto_refresh()
        info(LogRecord(
            event=LogEvent.OAUTH_AUTO_REFRESH_ALL_STARTED.value,
            message=f"Started auto-refresh for {len(oauth_manager.token_credentials)} loaded tokens"
        ))
    elif not auto_refresh_enabled:
        info(LogRecord(
            event=LogEvent.OAUTH_AUTO_REFRESH_DISABLED.value,
            message="Auto-refresh disabled by configuration"
        ))