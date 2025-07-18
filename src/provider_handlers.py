"""
Provider处理器抽象层
定义不同计费模式的处理器接口和实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import threading
import uuid
from .provider_types import ProviderConfig, BillingModel


@dataclass
class RequestContext:
    """请求上下文"""
    request_id: str
    original_request: Dict[str, Any]
    provider: ProviderConfig
    timestamp: float = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class BaseProviderHandler(ABC):
    """Provider处理器基类"""
    
    def __init__(self, provider: ProviderConfig):
        self.provider = provider
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_request_time = 0
        
    @abstractmethod
    async def prepare_request(self, context: RequestContext) -> Dict[str, Any]:
        """准备请求数据"""
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """处理响应数据"""
        pass
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: RequestContext) -> bool:
        """处理错误，返回是否需要重试"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider_name": self.provider.name,
            "billing_model": self.provider.get_billing_model().value,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / max(1, self.request_count),
            "last_request_time": self.last_request_time
        }


class TokenBasedProviderHandler(BaseProviderHandler):
    """Token-based provider处理器"""
    
    async def prepare_request(self, context: RequestContext) -> Dict[str, Any]:
        """直接转发请求，不需要session管理"""
        self.request_count += 1
        self.last_request_time = time.time()
        return context.original_request
    
    async def process_response(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """直接返回响应"""
        self.success_count += 1
        self.provider.mark_success()
        return response
    
    async def handle_error(self, error: Exception, context: RequestContext) -> bool:
        """处理错误"""
        self.failure_count += 1
        self.provider.mark_failure()
        
        # 对于token-based provider，大多数错误都不需要重试
        # 重试逻辑由provider manager的failover处理
        return False


@dataclass
class SessionState:
    """Session状态"""
    session_id: str
    thread_id: Optional[str] = None
    created_at: float = 0
    last_used_at: float = 0
    tool_calls_count: int = 0
    context_tokens: int = 0
    request_count: int = 0
    is_active: bool = True
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_used_at:
            self.last_used_at = self.created_at
    
    def is_expired(self, ttl: int) -> bool:
        """检查session是否过期"""
        return time.time() - self.last_used_at > ttl
    
    def should_rotate(self, config) -> bool:
        """检查是否需要轮转session"""
        # 基于上下文大小
        if self.context_tokens > config.max_context_tokens * config.auto_rotate_threshold:
            return True
        
        # 基于TTL
        if self.is_expired(config.session_ttl):
            return True
            
        # 基于工具调用数量（仅normal模式）
        if config.default_mode == "normal" and self.tool_calls_count >= config.max_tool_calls_per_session:
            return True
            
        return False


class SessionBasedProviderHandler(BaseProviderHandler):
    """Session-based provider处理器"""
    
    def __init__(self, provider: ProviderConfig):
        super().__init__(provider)
        self.session_state: Optional[SessionState] = None
        self.lock = threading.Lock()
        
        if not provider.session_config:
            from .provider_types import SessionConfig
            provider.session_config = SessionConfig()
    
    async def prepare_request(self, context: RequestContext) -> Dict[str, Any]:
        """准备session格式的请求"""
        self.request_count += 1
        self.last_request_time = time.time()
        
        with self.lock:
            # 获取或创建session
            session = await self._get_or_create_session()
            
            # 生成新的prompt_id
            prompt_id = str(uuid.uuid4())
            
            # 构建session请求
            session_request = {
                "session_id": session.session_id,
                "thread_id": session.thread_id,
                "prompt_id": prompt_id,
                "intent": context.original_request.get("intent", "user_prompt"),
                "mode": self._select_mode(context.original_request),
                "provider": "anthropic",  # 后端provider
                "provider_request": self._extract_provider_request(context.original_request)
            }
            
            # 更新session状态
            session.last_used_at = time.time()
            session.request_count += 1
            
            return session_request
    
    async def process_response(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """处理session响应"""
        self.success_count += 1
        
        with self.lock:
            if self.session_state:
                # 更新session统计
                self.session_state.last_used_at = time.time()
                
                # 估算token使用量和工具调用
                self._update_session_usage(response)
        
        # 提取实际的AI响应
        actual_response = self._extract_actual_response(response)
        return actual_response
    
    async def handle_error(self, error: Exception, context: RequestContext) -> bool:
        """处理session错误"""
        self.failure_count += 1
        
        # 分析错误类型
        error_type = self._classify_error(error)
        
        with self.lock:
            if error_type in ["thread_expired", "context_overflow", "session_invalid"]:
                # 需要轮转session
                await self._rotate_session(error_type)
                return True  # 重试
            elif error_type in ["tool_calls_limit", "user_interaction_required"]:
                # 只需要新prompt_id，不需要轮转session
                return True  # 重试
            else:
                # 其他错误不重试
                return False
    
    async def _get_or_create_session(self) -> SessionState:
        """获取或创建session"""
        if not self.session_state or self.session_state.should_rotate(self.provider.session_config):
            await self._rotate_session("auto_rotation")
        
        return self.session_state
    
    async def _rotate_session(self, reason: str = "manual"):
        """轮转到新session"""
        # 如果有旧session，可以考虑获取对话总结
        old_session = self.session_state
        summary = None
        
        if old_session and old_session.request_count > 0:
            summary = await self._summarize_session(old_session)
        
        # 创建新session
        new_session = SessionState(
            session_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4()) if self.provider.type.value == "zed" else None
        )
        
        self.session_state = new_session
        
        # 记录轮转日志
        print(f"[SESSION] Rotated session for {self.provider.name}, reason: {reason}")
        
        return new_session
    
    async def _summarize_session(self, session: SessionState) -> Optional[str]:
        """总结session内容（可选实现）"""
        # 这里可以调用总结API来保存对话历史
        # 暂时返回None，后续可以实现
        return None
    
    def _select_mode(self, request: Dict[str, Any]) -> str:
        """选择session模式"""
        # 简单逻辑：根据工具数量选择
        tools = request.get("tools", [])
        if len(tools) > 15:  # 大量工具调用，可能需要burn mode
            return "burn"
        return self.provider.session_config.default_mode
    
    def _extract_provider_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """提取provider请求部分"""
        # 移除session相关的字段，保留AI请求字段
        provider_request = {
            "model": request.get("model"),
            "messages": request.get("messages", []),
            "max_tokens": request.get("max_tokens", 8192),
            "temperature": request.get("temperature"),
            "top_p": request.get("top_p"),
            "tools": request.get("tools", []),
            "tool_choice": request.get("tool_choice")
        }
        
        # 移除None值
        return {k: v for k, v in provider_request.items() if v is not None}
    
    def _extract_actual_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """提取实际的AI响应"""
        # 如果response包含provider_response字段，提取它
        if "provider_response" in response:
            return response["provider_response"]
        
        # 否则直接返回整个response
        return response
    
    def _update_session_usage(self, response: Dict[str, Any]):
        """更新session使用统计"""
        if not self.session_state:
            return
            
        # 估算token使用量
        usage = response.get("usage", {})
        if usage:
            self.session_state.context_tokens += usage.get("total_tokens", 0)
        
        # 估算工具调用数量
        if "tool_calls" in response:
            tool_calls = response.get("tool_calls", [])
            self.session_state.tool_calls_count += len(tool_calls)
    
    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        error_msg = str(error).lower()
        
        # Session轮转错误
        if any(keyword in error_msg for keyword in ["context_length", "thread_expired", "session_invalid"]):
            return "thread_expired"
        
        # 工具调用限制错误
        if "tool_calls_limit" in error_msg or "25" in error_msg:
            return "tool_calls_limit"
        
        # 用户交互需求
        if "user_interaction" in error_msg:
            return "user_interaction_required"
        
        return "unknown_error"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        base_stats = super().get_stats()
        
        if self.session_state:
            base_stats.update({
                "session_id": self.session_state.session_id,
                "session_age": time.time() - self.session_state.created_at,
                "session_requests": self.session_state.request_count,
                "session_tool_calls": self.session_state.tool_calls_count,
                "session_context_tokens": self.session_state.context_tokens
            })
        
        return base_stats


class ProviderHandlerFactory:
    """Provider处理器工厂"""
    
    @staticmethod
    def create_handler(provider: ProviderConfig) -> BaseProviderHandler:
        """创建相应的处理器"""
        if provider.is_session_based():
            return SessionBasedProviderHandler(provider)
        else:
            return TokenBasedProviderHandler(provider)