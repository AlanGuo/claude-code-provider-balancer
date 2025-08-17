"""
认证管理器模块

负责处理各种认证方式：API Key、OAuth、透传等
从 ProviderManager 中分离出来，专注于认证相关逻辑
"""

from typing import Dict, Optional, Protocol
from enum import Enum
from urllib.parse import urlparse

from utils import debug, LogRecord, LogEvent
from utils.logging.formatters import mask_sensitive_data

class AuthType(str, Enum):
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ProviderProtocol(Protocol):
    """Provider协议 - 定义认证管理器需要的Provider接口"""
    name: str
    type: ProviderType
    auth_type: AuthType
    auth_value: str
    account_email: Optional[str]
    base_url: str


class ProviderAuth:
    """认证管理器 - 专门处理Provider认证逻辑"""
    
    def __init__(self):
        pass
    
    def get_provider_headers(self, provider: ProviderProtocol, original_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """获取Provider的认证头部，可选择性合并原始头部"""
        headers = {}
        
        debug(LogRecord(
            event=LogEvent.GET_PROVIDER_HEADERS_START.value,
            message=f"Provider {provider.name}: auth_type={provider.auth_type}, auth_value=[REDACTED]"
        ))
        
        # 打印原始请求头（在现有debug之后）
        if original_headers:
            debug(LogRecord(
                event=LogEvent.ORIGINAL_REQUEST_HEADERS_RECEIVED.value,
                message=f"Original headers for provider {provider.name}",
                data={
                    "original_headers": mask_sensitive_data({k: v for k, v in original_headers.items()}),
                    "provider": provider.name
                }
            ))
        
        # 复制原始请求头（排除需要替换的认证头和content-length头）
        if original_headers:
            headers.update(self._filter_original_headers(original_headers))
        
        # 添加Host头部，从provider的base_url中提取
        self._add_host_header(headers, provider)
        
        # 根据认证模式设置认证头部
        if provider.auth_value == "passthrough":
            self._handle_passthrough_auth(headers, provider, original_headers)
        else:
            self._handle_standard_auth(headers, provider)
        
        # 确保有Content-Type头部（如果原始请求没有的话）
        if not any(key.lower() == 'content-type' for key in headers.keys()):
            headers["content-type"] = "application/json"
        
        # 在return之前添加最终请求头打印
        debug(LogRecord(
            event=LogEvent.FINAL_PROVIDER_HEADERS.value,
            message=f"Final headers for provider {provider.name}",
            data={
                "final_headers": mask_sensitive_data({k: v for k, v in headers.items()}),
                "provider": provider.name
            }
        ))
        
        return headers
    
    def _filter_original_headers(self, original_headers: Dict[str, str]) -> Dict[str, str]:
        """过滤原始头部，移除需要替换的认证相关头部"""
        filtered = {}
        excluded_headers = {'authorization', 'x-api-key', 'host'}
        
        for key, value in original_headers.items():
            if key.lower() not in excluded_headers:
                filtered[key] = value
        
        return filtered
    
    def _add_host_header(self, headers: Dict[str, str], provider: ProviderProtocol):
        """从provider的base_url中提取host并添加到headers"""
        parsed_url = urlparse(provider.base_url)
        if parsed_url.hostname:
            host = parsed_url.hostname
            if parsed_url.port:
                host = f"{host}:{parsed_url.port}"
            headers["host"] = host
    
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
    
    
    def _handle_standard_auth(self, headers: Dict[str, str], provider: ProviderProtocol):
        """处理标准认证模式（API Key、Auth Token）"""
        # 获取实际的认证值
        auth_value = self._get_auth_value(provider)
        
        if provider.auth_type == AuthType.API_KEY:
            if provider.type == ProviderType.ANTHROPIC:
                headers["x-api-key"] = auth_value
            else:  # OpenAI compatible
                headers["Authorization"] = f"Bearer {auth_value}"
        elif provider.auth_type == AuthType.AUTH_TOKEN:
            # 对于使用auth_token的服务商
            headers["Authorization"] = f"Bearer {auth_value}"
            if provider.type == ProviderType.ANTHROPIC:
                # 对于Claude Code Official，需要特殊处理头部以确保OAuth兼容性
                if provider.name == "Claude Code Official" and provider.auth_value == "oauth":
                    self._apply_claude_official_headers_fix(headers)
    
    def _apply_claude_official_headers_fix(self, headers: Dict[str, str]):
        """为Claude Code Official应用头部修正，确保OAuth兼容性"""
        # 确保有oauth-2025-04-20 beta标识，这是成功认证的关键
        anthropic_beta = headers.get("anthropic-beta", "")
        
        # 添加oauth-2025-04-20如果没有的话
        if "oauth-2025-04-20" not in anthropic_beta:
            if anthropic_beta:
                # 如果已有其他beta标识，添加到前面
                headers["anthropic-beta"] = f"oauth-2025-04-20,{anthropic_beta}"
            else:
                # 如果没有beta标识，只添加OAuth相关的
                headers["anthropic-beta"] = "oauth-2025-04-20"

        debug(LogRecord(
            event=LogEvent.CLAUDE_OFFICIAL_HEADERS_APPLIED.value,
            message=f"Applied Claude Official OAuth headers fix",
            data={"anthropic_beta": headers.get("anthropic-beta")}
        ))
    
    def _get_auth_value(self, provider: ProviderProtocol) -> str:
        """获取实际的认证值，如果是OAuth则从keyring获取"""
        if provider.auth_value == "oauth":
            # 从OAuth manager获取token
            oauth_manager = self._get_oauth_manager()
            
            debug(LogRecord(
                event=LogEvent.OAUTH_MANAGER_CHECK.value, 
                message=f"OAuth manager obtained for {provider.name}: {oauth_manager is not None}"
            ))
            
            if not oauth_manager:
                # OAuth manager未初始化，触发OAuth授权流程
                self._trigger_oauth_authorization(provider)
            
            # 如果provider有指定account_email，则获取对应账户的token
            if hasattr(provider, 'account_email') and provider.account_email:
                access_token = oauth_manager.get_token_by_email(provider.account_email)
                debug(LogRecord(
                    event=LogEvent.OAUTH_TOKEN_USED_BY_EMAIL.value,
                    message=f"Requesting OAuth token for account {provider.account_email} from provider {provider.name}"
                ))
            else:
                # 否则使用轮询机制获取token
                access_token = oauth_manager.get_current_token()
                debug(LogRecord(
                    event=LogEvent.OAUTH_TOKEN_USED.value,
                    message=f"Using round-robin OAuth token for provider {provider.name}"
                ))
            
            if not access_token:
                # 触发OAuth授权流程
                self._trigger_oauth_authorization(provider)
            
            return access_token
        else:
            # 直接返回配置中的auth_value
            debug(LogRecord(
                event=LogEvent.GET_PROVIDER_HEADERS_START.value,
                message=f"Using configured auth_value for {provider.name} (non-oauth)"
            ))
            return provider.auth_value
    
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
    
    def handle_oauth_authorization_required(self, provider: ProviderProtocol, http_status_code: int = 401):
        """处理OAuth授权需求的用户交互"""
        if provider.name == "Claude Code Official":
            # Check if OAuth manager is available
            oauth_manager = self._get_oauth_manager()
            
            if not oauth_manager:
                self._print_oauth_manager_unavailable()
            
            # Print authorization instructions
            self._print_oauth_authorization_instructions(http_status_code, provider)

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
    
    def _print_oauth_authorization_instructions(self, http_status_code: int, provider: Optional[ProviderProtocol] = None):
        """打印OAuth授权指令"""
        print("\n" + "="*80)
        if http_status_code == 403:
            print("🔒 FORBIDDEN ACCESS - OAUTH AUTHENTICATION REQUIRED")
        else:  # 401
            print("🔐 AUTHENTICATION REQUIRED - OAUTH LOGIN NEEDED")
        
        if provider and hasattr(provider, 'account_email') and provider.account_email:
            print(f"👤 Required account: {provider.account_email}")
        
        print("="*80)
        print()
        print("To continue using Claude Code Provider Balancer, you need to:")
        print()
        print("1. 🌐 Open this URL in your browser:")
        print("   http://localhost:9090/oauth/generate-url")
        print()
        print("2. 🔑 Sign in with your Claude Code account")
        
        if provider and hasattr(provider, 'account_email') and provider.account_email:
            print(f"   ⚠️  Make sure to use account: {provider.account_email}")
        
        print()
        print("3. ✅ Grant permission to the application")
        print()
        print("4. 🔄 The token will be saved automatically")
        print()
        print("5. ⚡ Retry your request - it should work now!")
        print()
        print("="*80)
        print()