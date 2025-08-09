"""
认证管理器模块

负责处理各种认证方式：API Key、OAuth、透传等
从 ProviderManager 中分离出来，专注于认证相关逻辑
"""

from typing import Dict, Optional, Protocol
from enum import Enum

from utils import debug, LogRecord, LogEvent


class AuthType(str, Enum):
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"
    OAUTH = "oauth"


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ProviderProtocol(Protocol):
    """Provider协议 - 定义认证管理器需要的Provider接口"""
    name: str
    type: ProviderType
    auth_type: AuthType
    auth_value: str


class ProviderAuth:
    """认证管理器 - 专门处理Provider认证逻辑"""
    
    def __init__(self):
        pass
    
    def get_provider_headers(self, provider: ProviderProtocol, original_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """获取Provider的认证头部，可选择性合并原始头部"""
        headers = {
            "Content-Type": "application/json"
        }
        
        debug(LogRecord(
            event=LogEvent.GET_PROVIDER_HEADERS_START.value,
            message=f"Provider {provider.name}: auth_type={provider.auth_type}, auth_value=[REDACTED]"
        ))
        
        # 复制原始请求头（排除需要替换的认证头、host头和content-length头）
        if original_headers:
            headers.update(self._filter_original_headers(original_headers))
        
        # 根据认证模式设置认证头部
        if provider.auth_value == "passthrough":
            self._handle_passthrough_auth(headers, provider, original_headers)
        elif provider.auth_type == AuthType.OAUTH:
            self._handle_oauth_auth(headers, provider)
        else:
            self._handle_standard_auth(headers, provider)
        
        return headers
    
    def _filter_original_headers(self, original_headers: Dict[str, str]) -> Dict[str, str]:
        """过滤原始头部，移除需要替换的认证相关头部"""
        filtered = {}
        excluded_headers = {'authorization', 'x-api-key', 'host', 'content-length'}
        
        for key, value in original_headers.items():
            if key.lower() not in excluded_headers:
                filtered[key] = value
        
        return filtered
    
    def _handle_passthrough_auth(self, headers: Dict[str, str], provider: ProviderProtocol, original_headers: Optional[Dict[str, str]]):
        """处理透传认证模式"""
        if not original_headers:
            return
            
        # 保留原始请求的认证头部（不区分大小写查找）
        for key, value in original_headers.items():
            key_lower = key.lower()
            if key_lower == "authorization":
                headers["Authorization"] = value
            elif key_lower == "x-api-key":
                headers["x-api-key"] = value
        
        # 为Anthropic类型的provider添加版本头
        if provider.type == ProviderType.ANTHROPIC:
            headers["anthropic-version"] = "2023-06-01"
    
    def _handle_oauth_auth(self, headers: Dict[str, str], provider: ProviderProtocol):
        """处理OAuth认证模式"""
        # 获取OAuth manager
        oauth_manager = self._get_oauth_manager()
        
        if not oauth_manager:
            # OAuth manager未初始化，触发OAuth授权流程
            self._trigger_oauth_authorization(provider)
        
        access_token = oauth_manager.get_current_token()
        if not access_token:
            # 触发OAuth授权流程
            self._trigger_oauth_authorization(provider)
        
        # 使用OAuth token作为Bearer token
        headers["Authorization"] = f"Bearer {access_token}"
        
        # 为Anthropic类型的provider添加版本头
        if provider.type == ProviderType.ANTHROPIC:
            headers["anthropic-version"] = "2023-06-01"
    
    def _handle_standard_auth(self, headers: Dict[str, str], provider: ProviderProtocol):
        """处理标准认证模式（API Key、Auth Token）"""
        if provider.auth_type == AuthType.API_KEY:
            if provider.type == ProviderType.ANTHROPIC:
                headers["x-api-key"] = provider.auth_value
                headers["anthropic-version"] = "2023-06-01"
            else:  # OpenAI compatible
                headers["Authorization"] = f"Bearer {provider.auth_value}"
        elif provider.auth_type == AuthType.AUTH_TOKEN:
            # 对于使用auth_token的服务商
            headers["Authorization"] = f"Bearer {provider.auth_value}"
            if provider.type == ProviderType.ANTHROPIC:
                headers["anthropic-version"] = "2023-06-01"
    
    def _get_oauth_manager(self):
        """获取OAuth管理器"""
        try:
            from oauth import get_oauth_manager
            oauth_manager = get_oauth_manager()
            debug(LogRecord(
                event=LogEvent.OAUTH_MANAGER_CHECK.value, 
                message=f"OAuth manager status: {oauth_manager is not None}, type: {type(oauth_manager)}"
            ))
            return oauth_manager
        except ImportError:
            return None
    
    def _trigger_oauth_authorization(self, provider: ProviderProtocol):
        """触发OAuth授权流程并抛出401错误"""
        self.handle_oauth_authorization_required(provider)
        
        # 创建一个401错误来触发标准的错误处理流程
        from httpx import HTTPStatusError
        import httpx
        response = httpx.Response(
            status_code=401,
            text="Unauthorized: OAuth token not available",
            request=httpx.Request("POST", "http://example.com")
        )
        raise HTTPStatusError("401 Unauthorized", request=response.request, response=response)
    
    def handle_oauth_authorization_required(self, provider: ProviderProtocol, http_status_code: int = 401) -> str:
        """处理OAuth授权需求的用户交互"""
        if provider.name == "Claude Code Official":
            # Check if OAuth manager is available
            oauth_manager = self._get_oauth_manager()
            
            if not oauth_manager:
                self._print_oauth_manager_unavailable()
                return ""
            
            # Get authorization URL from OAuth manager
            auth_url = oauth_manager.generate_login_url()
            if not auth_url:
                self._print_oauth_setup_failed()
                return ""
            
            # Print authorization instructions
            self._print_oauth_authorization_instructions(auth_url, http_status_code)
            return auth_url
        
        return ""
    
    def _print_oauth_manager_unavailable(self):
        """打印OAuth管理器不可用的提示"""
        print("\n" + "="*80)
        print("❌ OAUTH MANAGER NOT AVAILABLE")
        print("="*80)
        print("The OAuth manager failed to initialize properly.")
        print("Please check the logs for initialization errors.")
        print("OAuth authentication is not available at this time.")
        print("="*80)
        print()
    
    def _print_oauth_setup_failed(self):
        """打印OAuth设置失败的提示"""
        print("\n" + "="*80)
        print("❌ OAUTH SETUP FAILED")
        print("="*80)
        print("Failed to get authorization URL from OAuth manager.")
        print("OAuth authentication cannot proceed at this time.")
        print("="*80)
        print()
    
    def _print_oauth_authorization_instructions(self, auth_url: str, http_status_code: int):
        """打印OAuth授权指令"""
        # Note: auth_url parameter kept for interface compatibility but not used
        # as we now direct users to the local URL generator
        print("\n" + "="*80)
        if http_status_code == 403:
            print("🔒 FORBIDDEN ACCESS - OAUTH AUTHENTICATION REQUIRED")
        else:  # 401
            print("🔐 AUTHENTICATION REQUIRED - OAUTH LOGIN NEEDED")
        print("="*80)
        print()
        print("To continue using Claude Code Provider Balancer, you need to:")
        print()
        print("1. 🌐 Open this URL in your browser:")
        print("   http://localhost:9090/oauth/generate-url")
        print()
        print("2. 🔑 Sign in with your Claude Code account")
        print()
        print("3. ✅ Grant permission to the application")
        print()
        print("4. 🔄 The token will be saved automatically")
        print()
        print("5. ⚡ Retry your request - it should work now!")
        print()
        print("="*80)
        print()