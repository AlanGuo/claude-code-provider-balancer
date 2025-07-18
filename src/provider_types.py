"""
Provider类型定义和枚举
定义不同计费模式的provider类型
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class BillingModel(str, Enum):
    """计费模式"""
    TOKEN_BASED = "token_based"
    SESSION_BASED = "session_based"


class ProviderType(str, Enum):
    """Provider类型"""
    # Token-based providers
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    
    # Session-based providers  
    ZED = "zed"


class AuthType(str, Enum):
    """认证类型"""
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"


@dataclass
class SessionConfig:
    """Session-based provider配置"""
    max_context_tokens: int = 120000
    max_tool_calls_per_session: int = 25
    session_ttl: int = 3600  # seconds
    default_mode: str = "normal"  # normal | burn
    auto_rotate_threshold: float = 0.8
    
    # 模式配置
    modes: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "normal": {
            "cost_per_prompt": 0.04,
            "max_tool_calls": 25
        },
        "burn": {
            "cost_per_request": 0.05,
            "max_tool_calls": -1  # unlimited
        }
    })


@dataclass  
class ProviderConfig:
    """统一的Provider配置"""
    name: str
    type: ProviderType
    base_url: str
    auth_type: AuthType
    auth_value: str
    enabled: bool = True
    proxy: Optional[str] = None
    
    # 计费模式（可选，会自动推断）
    billing_model: Optional[BillingModel] = None
    
    # Session-based provider配置
    session_config: Optional[SessionConfig] = None
    
    # Token-based provider状态
    failure_count: int = 0
    last_failure_time: float = 0
    
    def get_billing_model(self) -> BillingModel:
        """获取计费模式，如果未指定则自动推断"""
        if self.billing_model:
            return self.billing_model
            
        # 自动推断
        if self.type in [ProviderType.ZED]:
            return BillingModel.SESSION_BASED
        else:
            return BillingModel.TOKEN_BASED
    
    def is_session_based(self) -> bool:
        """判断是否为session-based provider"""
        return self.get_billing_model() == BillingModel.SESSION_BASED
    
    def is_token_based(self) -> bool:
        """判断是否为token-based provider"""
        return self.get_billing_model() == BillingModel.TOKEN_BASED
    
    def is_healthy(self, cooldown_seconds: int = 60) -> bool:
        """检查provider是否健康（仅对token-based有效）"""
        if self.is_session_based():
            return True  # session-based provider健康状态由session管理器管理
            
        if self.failure_count == 0:
            return True
            
        import time
        return time.time() - self.last_failure_time > cooldown_seconds
    
    def mark_failure(self):
        """标记失败（仅对token-based有效）"""
        if self.is_token_based():
            import time
            self.failure_count += 1
            self.last_failure_time = time.time()
    
    def mark_success(self):
        """标记成功（仅对token-based有效）"""
        if self.is_token_based():
            self.failure_count = 0
            self.last_failure_time = 0