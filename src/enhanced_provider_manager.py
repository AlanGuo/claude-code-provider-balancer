"""
Enhanced Provider Manager
集成了智能路由和session-based provider支持的增强版provider管理器
"""

import os
import time
import yaml
import threading
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from .provider_types import ProviderConfig, ProviderType, AuthType, BillingModel, SessionConfig
from .provider_handlers import BaseProviderHandler, ProviderHandlerFactory, RequestContext
from .intelligent_router import SimpleRouter, IntelligentProviderSelector, RouteDecision
from .provider_manager import SelectionStrategy, ModelRoute  # 保持兼容性


class EnhancedProviderManager:
    """增强版Provider管理器，支持智能路由和session-based providers"""
    
    def __init__(self, config_path: str = "providers.yaml"):
        # 配置文件路径
        if not os.path.isabs(config_path):
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / config_path
        
        self.config_path = Path(config_path)
        
        # 核心组件
        self.providers: List[ProviderConfig] = []
        self.provider_handlers: Dict[str, BaseProviderHandler] = {}
        self.settings: Dict[str, Any] = {}
        self.model_routes: Dict[str, List[ModelRoute]] = {}
        
        # 智能路由组件
        self.intelligent_router: Optional[SimpleRouter] = None
        self.provider_selector: Optional[IntelligentProviderSelector] = None
        self.routing_enabled = False
        
        # 兼容性支持
        self.selection_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
        self._round_robin_indices: Dict[str, int] = {}
        self._last_request_time: float = 0
        self._last_successful_provider: Optional[str] = None
        self._idle_recovery_interval: float = 300
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 加载配置
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 加载基础设置
            self.settings = config.get('settings', {})
            self.selection_strategy = SelectionStrategy(
                self.settings.get('selection_strategy', 'priority')
            )
            self._idle_recovery_interval = self.settings.get('idle_recovery_interval', 300)
            
            # 加载路由配置
            routing_config = config.get('routing', {})
            self.routing_enabled = routing_config.get('enabled', False)
            
            if self.routing_enabled:
                self.intelligent_router = SimpleRouter(routing_config)
                self.provider_selector = IntelligentProviderSelector(self.intelligent_router)
            
            # 加载providers
            self._load_providers(config.get('providers', []))
            
            # 加载模型路由
            self._load_model_routes(config.get('model_routes', {}))
            
            if not self.providers:
                raise ValueError("No enabled providers found in configuration")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load provider configuration: {e}")
    
    def _load_providers(self, providers_config: List[Dict[str, Any]]):
        """加载providers配置"""
        self.providers = []
        self.provider_handlers = {}
        
        for provider_config in providers_config:
            if not provider_config.get('enabled', True):
                continue
                
            # 创建provider配置
            provider = ProviderConfig(
                name=provider_config['name'],
                type=ProviderType(provider_config['type']),
                base_url=provider_config['base_url'],
                auth_type=AuthType(provider_config['auth_type']),
                auth_value=provider_config['auth_value'],
                enabled=provider_config.get('enabled', True),
                proxy=provider_config.get('proxy'),
                billing_model=BillingModel(provider_config['billing_model']) if 'billing_model' in provider_config else None
            )
            
            # 加载session配置（如果是session-based provider）
            if provider.is_session_based():
                session_config_data = provider_config.get('session_config', {})
                provider.session_config = SessionConfig(
                    max_context_tokens=session_config_data.get('max_context_tokens', 120000),
                    max_tool_calls_per_session=session_config_data.get('max_tool_calls_per_session', 25),
                    session_ttl=session_config_data.get('session_ttl', 3600),
                    default_mode=session_config_data.get('default_mode', 'normal'),
                    auto_rotate_threshold=session_config_data.get('auto_rotate_threshold', 0.8),
                    modes=session_config_data.get('modes', {})
                )
            
            self.providers.append(provider)
            
            # 创建对应的处理器
            handler = ProviderHandlerFactory.create_handler(provider)
            self.provider_handlers[provider.name] = handler
    
    def _load_model_routes(self, routes_config: Dict[str, Any]):
        """加载模型路由配置"""
        self.model_routes = {}
        
        for model_pattern, routes in routes_config.items():
            route_list = []
            for route_config in routes:
                if isinstance(route_config, dict):
                    route = ModelRoute(
                        provider=route_config['provider'],
                        model=route_config['model'],
                        priority=route_config['priority'],
                        enabled=route_config.get('enabled', True)
                    )
                    route_list.append(route)
            self.model_routes[model_pattern] = route_list
    
    async def process_request(self, request: Dict[str, Any], requested_model: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        处理请求的主要入口点
        
        Args:
            request: 原始请求
            requested_model: 请求的模型名称
            
        Returns:
            (处理后的请求, 选中的provider名称, 路由信息)
        """
        with self.lock:
            self.mark_request_start()
            
            # 选择provider
            if self.routing_enabled:
                providers, route_info = await self._intelligent_select_providers(request, requested_model)
            else:
                providers = self._legacy_select_providers(requested_model)
                route_info = {'method': 'legacy', 'providers_count': len(providers)}
            
            if not providers:
                raise RuntimeError(f"No available providers for model: {requested_model}")
            
            # 选择第一个可用的provider
            selected_provider = providers[0]
            
            # 获取处理器并处理请求
            handler = self.provider_handlers[selected_provider.name]
            context = RequestContext(
                request_id=f"req_{int(time.time() * 1000)}",
                original_request=request,
                provider=selected_provider
            )
            
            processed_request = await handler.prepare_request(context)
            
            return processed_request, selected_provider.name, route_info
    
    async def _intelligent_select_providers(self, request: Dict[str, Any], requested_model: str) -> Tuple[List[ProviderConfig], Dict[str, Any]]:
        """智能选择providers"""
        # 获取模型对应的所有providers
        all_providers = self._get_providers_for_model(requested_model)
        
        if not all_providers:
            return [], {'error': f'No providers configured for model: {requested_model}'}
        
        # 使用智能路由选择
        selected_providers, route_info = self.provider_selector.select_providers(request, all_providers)
        
        # 过滤健康的providers
        healthy_providers = [p for p in selected_providers if self._is_provider_healthy(p)]
        
        route_info.update({
            'method': 'intelligent',
            'total_configured': len(all_providers),
            'healthy_count': len(healthy_providers)
        })
        
        return healthy_providers, route_info
    
    def _legacy_select_providers(self, requested_model: str) -> List[ProviderConfig]:
        """传统的provider选择方法（兼容性）"""
        # 使用原有的选择逻辑
        options = self.select_model_and_provider_options(requested_model)
        return [option[1] for option in options]
    
    def _get_providers_for_model(self, requested_model: str) -> List[ProviderConfig]:
        """获取支持指定模型的providers"""
        # 使用现有的模型路由逻辑
        model_options = self.select_model_and_provider_options(requested_model)
        return [option[1] for option in model_options]
    
    def _is_provider_healthy(self, provider: ProviderConfig) -> bool:
        """检查provider是否健康"""
        if not provider.enabled:
            return False
        
        # Session-based providers的健康状态由handler管理
        if provider.is_session_based():
            return True
        
        # Token-based providers使用传统的健康检查
        return provider.is_healthy(self.get_failure_cooldown())
    
    async def handle_response(self, response: Dict[str, Any], provider_name: str, context: RequestContext) -> Dict[str, Any]:
        """处理响应"""
        handler = self.provider_handlers.get(provider_name)
        if not handler:
            return response
        
        # 标记成功
        self.mark_provider_success(provider_name)
        
        # 处理响应
        return await handler.process_response(response, context)
    
    async def handle_error(self, error: Exception, provider_name: str, context: RequestContext) -> bool:
        """处理错误，返回是否需要重试"""
        handler = self.provider_handlers.get(provider_name)
        if not handler:
            return False
        
        return await handler.handle_error(error, context)
    
    def get_provider_headers(self, provider: ProviderConfig, original_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """获取provider的认证headers"""
        headers = {"Content-Type": "application/json"}
        
        if original_headers:
            for key, value in original_headers.items():
                if key.lower() not in ['authorization', 'x-api-key', 'host', 'content-length']:
                    headers[key] = value
        
        # 检查是否使用passthrough模式
        if provider.auth_value == "passthrough":
            if original_headers:
                for key, value in original_headers.items():
                    if key.lower() == "authorization":
                        headers["Authorization"] = value
                    elif key.lower() == "x-api-key":
                        headers["x-api-key"] = value
            if provider.type == ProviderType.ANTHROPIC:
                headers["anthropic-version"] = "2023-06-01"
        else:
            # 正常认证模式
            if provider.auth_type == AuthType.API_KEY:
                if provider.type == ProviderType.ANTHROPIC:
                    headers["x-api-key"] = provider.auth_value
                    headers["anthropic-version"] = "2023-06-01"
                else:
                    headers["Authorization"] = f"Bearer {provider.auth_value}"
            elif provider.auth_type == AuthType.AUTH_TOKEN:
                headers["Authorization"] = f"Bearer {provider.auth_value}"
                if provider.type == ProviderType.ANTHROPIC:
                    headers["anthropic-version"] = "2023-06-01"
        
        return headers
    
    def get_request_url(self, provider: ProviderConfig, endpoint: str) -> str:
        """获取请求URL"""
        base_url = provider.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/{endpoint}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        status = {
            "total_providers": len(self.providers),
            "healthy_providers": len([p for p in self.providers if self._is_provider_healthy(p)]),
            "routing_enabled": self.routing_enabled,
            "selection_strategy": self.selection_strategy.value,
            "total_model_routes": len(self.model_routes),
            "providers": []
        }
        
        for provider in self.providers:
            provider_stats = {
                "name": provider.name,
                "type": provider.type.value,
                "billing_model": provider.get_billing_model().value,
                "base_url": provider.base_url,
                "enabled": provider.enabled,
                "healthy": self._is_provider_healthy(provider),
                "proxy": provider.proxy
            }
            
            # 添加处理器统计
            handler = self.provider_handlers.get(provider.name)
            if handler:
                provider_stats.update(handler.get_stats())
            
            status["providers"].append(provider_stats)
        
        return status
    
    def reload_config(self):
        """重新加载配置"""
        self.load_config()
    
    def shutdown(self):
        """关闭管理器"""
        # 清理资源
        self.provider_handlers.clear()
    
    # 兼容性方法 - 保持与原有代码的兼容性
    def select_model_and_provider_options(self, requested_model: str) -> List[Tuple[str, ProviderConfig]]:
        """兼容性方法：选择模型和provider选项"""
        # 1. 精确匹配
        if requested_model in self.model_routes:
            options = self._build_options_from_routes(self.model_routes[requested_model], requested_model)
            if options:
                return self._apply_selection_strategy(options, requested_model)
        
        # 2. 通配符匹配
        import re
        for pattern, routes in self.model_routes.items():
            if self._matches_pattern(requested_model, pattern):
                options = self._build_options_from_routes(routes, requested_model)
                if options:
                    return self._apply_selection_strategy(options, requested_model)
        
        return []
    
    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """检查模型名是否匹配模式"""
        model_lower = model_name.lower()
        pattern_lower = pattern.lower()
        
        if '*' in pattern:
            import re
            regex_pattern = pattern_lower.replace('*', '.*')
            return bool(re.search(regex_pattern, model_lower))
        else:
            return pattern_lower == model_lower
    
    def _build_options_from_routes(self, routes: List[ModelRoute], requested_model: str) -> List[Tuple[str, ProviderConfig, int]]:
        """从路由配置构建选项"""
        options = []
        cooldown = self.get_failure_cooldown()
        
        for route in routes:
            if not route.enabled:
                continue
                
            provider = self._get_provider_by_name(route.provider)
            if not provider or not provider.enabled or not provider.is_healthy(cooldown):
                continue
            
            target_model = route.model
            if target_model == "passthrough":
                target_model = requested_model
            
            options.append((target_model, provider, route.priority))
        
        return options
    
    def _apply_selection_strategy(self, options: List[Tuple[str, ProviderConfig, int]], requested_model: str) -> List[Tuple[str, ProviderConfig]]:
        """应用选择策略"""
        if not options:
            return []
        
        # 按优先级排序
        sorted_options = sorted(options, key=lambda x: x[2])
        return [(model, provider) for model, provider, priority in sorted_options]
    
    def _get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """根据名称获取provider"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
    
    def get_failure_cooldown(self) -> int:
        """获取失败冷却时间"""
        return self.settings.get('failure_cooldown', 60)
    
    def get_request_timeout(self) -> int:
        """获取请求超时时间"""
        return self.settings.get('request_timeout', 300)
    
    def mark_request_start(self):
        """标记请求开始"""
        self._last_request_time = time.time()
    
    def mark_provider_success(self, provider_name: str):
        """标记provider成功"""
        self._last_successful_provider = provider_name
        
        # 对于token-based providers，标记成功
        provider = self._get_provider_by_name(provider_name)
        if provider and provider.is_token_based():
            provider.mark_success()
    
    def get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """根据名称获取provider"""
        return self._get_provider_by_name(name)