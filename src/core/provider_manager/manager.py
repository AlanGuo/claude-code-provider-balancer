"""
Provider Manager for Claude Code Provider Balancer
Manages multiple Claude Code and OpenAI-compatible providers with simplified model routing.
"""

import os
import time
import yaml
import re
import random
import threading
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import httpx

# OAuth manager will be imported dynamically when needed
from utils import info, warning, error, debug, LogRecord, LogEvent
from .health import (
    get_error_handling_decision
)
from .provider_auth import ProviderAuth


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AuthType(str, Enum):
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"
    OAUTH = "oauth"


class SelectionStrategy(str, Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class StreamingMode(str, Enum):
    AUTO = "auto"  # Based on provider type (anthropic=direct, openai=background)
    DIRECT = "direct"  # Direct provider streaming without background collection
    BACKGROUND = "background"  # Background collection then streaming to client


@dataclass
class ModelRoute:
    provider: str
    model: str
    priority: int
    enabled: bool = True
    account_email: Optional[str] = None  # 可选的账户邮箱，用于区分相同name的providers


@dataclass
class Provider:
    name: str
    type: ProviderType
    base_url: str
    auth_type: AuthType
    auth_value: str
    enabled: bool = True
    proxy: Optional[str] = None
    streaming_mode: StreamingMode = StreamingMode.AUTO
    account_email: Optional[str] = None  # 新增字段，用于OAuth账户标识
    failure_count: int = 0
    last_failure_time: float = 0  # 保留作为统计指标
    last_unhealthy_time: float = 0  # 用于健康检查的时间戳
    last_success_time: float = 0  # 添加成功时间跟踪
    
    def is_healthy(self, cooldown_seconds: int = 60) -> bool:
        """Check if provider is healthy (not in unhealthy cooldown period)"""
        if self.last_unhealthy_time == 0:
            return True
        return time.time() - self.last_unhealthy_time > cooldown_seconds
    
    def mark_failure(self):
        """Mark provider as failed"""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def mark_success(self):
        """Mark provider as successful (reset failure count and unhealthy state)"""
        self.failure_count = 0
        self.last_failure_time = 0  # 保留作为统计指标
        self.last_unhealthy_time = 0  # 重置unhealthy状态
        self.last_success_time = time.time()  # 记录成功时间
    
    def get_effective_streaming_mode(self) -> StreamingMode:
        """Get the effective streaming mode based on configuration and provider type"""
        if self.streaming_mode == StreamingMode.AUTO:
            # Auto mode: anthropic providers use direct, openai providers use background
            if self.type == ProviderType.ANTHROPIC:
                return StreamingMode.DIRECT
            else:  # ProviderType.OPENAI
                return StreamingMode.BACKGROUND
        else:
            # Use explicitly configured mode
            return self.streaming_mode


class ProviderManager:
    def __init__(self, config_path: str = "config.yaml"):
        # Determine the absolute path to the config file
        if not os.path.isabs(config_path):
            # If relative path, look for it in project root (three levels up from src/core/provider_manager)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            config_path = project_root / config_path
        
        self.config_path = Path(config_path)
        self.providers: List[Provider] = []
        self.settings: Dict[str, Any] = {}
        
        # Provider认证处理器
        self.provider_auth = ProviderAuth()
        
        # 简化的配置
        self.model_routes: Dict[str, List[ModelRoute]] = {}
        self.selection_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
        
        # 用于round_robin策略的索引记录
        self._round_robin_indices: Dict[str, int] = {}
        
        # 请求活跃状态跟踪
        self._last_request_time: float = 0
        self._last_successful_provider: Optional[str] = None  # 记录最后成功的provider名称
        self._sticky_provider_duration: float = 300  # 默认粘滞provider持续时间为5分钟
        
        # OAuth配置
        self.oauth_auto_refresh_enabled: bool = True
        
        # 健康检查配置
        self.unhealthy_threshold: int = 2
        self.unhealthy_reset_on_success: bool = True 
        self.unhealthy_reset_timeout: float = 300  # 5分钟
        
        self.load_config()
    
    def load_config(self):
        """Load simplified configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.settings = config.get('settings', {})
            self.selection_strategy = SelectionStrategy(
                self.settings.get('selection_strategy', 'priority')
            )
            
            # 加载OAuth配置
            oauth_config = self.settings.get('oauth', {})
            self.oauth_auto_refresh_enabled = oauth_config.get('enable_auto_refresh', True)
            
            # 加载智能恢复配置
            self._sticky_provider_duration = self.settings.get('sticky_provider_duration', 300)
            
            # 加载健康检查配置
            self.unhealthy_threshold = self.settings.get('unhealthy_threshold', 2)
            self.unhealthy_reset_on_success = self.settings.get('unhealthy_reset_on_success', True)
            self.unhealthy_reset_timeout = self.settings.get('unhealthy_reset_timeout', 300)
            
            # 加载服务商配置
            providers_config = config.get('providers', [])
            self.providers = []
            
            for provider_config in providers_config:
                if provider_config.get('enabled', True):
                    # Parse streaming_mode with default to AUTO
                    streaming_mode_str = provider_config.get('streaming_mode', 'auto')
                    try:
                        streaming_mode = StreamingMode(streaming_mode_str)
                    except ValueError:
                        print(f"Warning: Invalid streaming_mode '{streaming_mode_str}' for provider '{provider_config['name']}', using 'auto'")
                        streaming_mode = StreamingMode.AUTO
                    
                    provider = Provider(
                        name=provider_config['name'],
                        type=ProviderType(provider_config['type']),
                        base_url=provider_config['base_url'],
                        auth_type=AuthType(provider_config['auth_type']),
                        auth_value=provider_config['auth_value'],
                        enabled=provider_config.get('enabled', True),
                        proxy=provider_config.get('proxy'),
                        streaming_mode=streaming_mode,
                        account_email=provider_config.get('account_email')  # 加载账户邮箱
                    )
                    debug(LogRecord(
                        event=LogEvent.PROVIDER_LOADED.value,
                        message=f"Loaded provider {provider.name} with auth_type={provider.auth_type}, auth_value=[REDACTED], account_email={provider.account_email}"
                    ))
                    self.providers.append(provider)
            
            # 加载模型路由配置
            self._load_model_routes(config.get('model_routes', {}))
            
            if not self.providers:
                raise ValueError("No enabled providers found in configuration")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load provider configuration: {e}")
    
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
                        enabled=route_config.get('enabled', True),
                        account_email=route_config.get('account_email')  # 支持account_email字段
                    )
                    route_list.append(route)
            self.model_routes[model_pattern] = route_list
    
    def _get_provider_by_name(self, name: str) -> Optional[Provider]:
        """根据名称获取服务商（返回第一个匹配的，保持向后兼容）"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
    
    def get_provider_by_name_and_account(self, name: str, account_email: Optional[str] = None) -> Optional[Provider]:
        """根据名称和账户邮箱获取服务商"""
        for provider in self.providers:
            if provider.name == name:
                # 如果指定了account_email，必须完全匹配（包括None）
                if account_email is not None:
                    if provider.account_email and provider.account_email.lower() == account_email.lower():
                        return provider
                # 如果没有指定account_email，优先返回也没有account_email的provider
                elif provider.account_email is None:
                    return provider
        
        # 如果没有找到精确匹配，且没有指定account_email，返回第一个匹配name的
        if account_email is None:
            return self._get_provider_by_name(name)
        
        return None
    
    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """检查模型名是否匹配给定的模式"""
        model_lower = model_name.lower()
        pattern_lower = pattern.lower()
        
        # 支持通配符匹配
        if '*' in pattern:
            regex_pattern = pattern_lower.replace('*', '.*')
            return bool(re.search(regex_pattern, model_lower))
        else:
            # 精确匹配
            return pattern_lower == model_lower
    
    def select_model_and_provider_options(self, requested_model: str, provider_name: Optional[str] = None) -> List[Tuple[str, Provider]]:
        """
        简化的模型选择逻辑
        返回按优先级排序的 (target_model, provider) 列表
        
        Args:
            requested_model: 请求的模型名称
            provider_name: 可选的指定provider名称，如果指定则只返回该provider的选项
        """
        # If provider is specified, return only that provider option
        if provider_name:
            # Find the specified provider
            target_provider = None
            for provider in self.providers:
                if provider.name == provider_name:
                    target_provider = provider
                    break
            
            if not target_provider:
                # Provider not found
                return []
                
            # Check if provider is healthy and enabled
            if not target_provider.enabled or not target_provider.is_healthy(self.get_failure_cooldown()):
                return []
            
            # Find model route for this specific provider
            # Look for the target model in the provider's configured models
            target_model = requested_model  # Default to passthrough
            
            # Check if there's a specific model mapping for this provider
            for pattern, routes in self.model_routes.items():
                if self._matches_pattern(requested_model, pattern):
                    for route in routes:
                        if route.provider == provider_name:
                            target_model = route.model if route.model != "passthrough" else requested_model
                            break
                    break
            
            return [(target_model, target_provider)]
        
        # Default behavior: return all available options for failover
        # 1. 精确匹配
        if requested_model in self.model_routes:
            options = self._build_options_from_routes(self.model_routes[requested_model], requested_model)
            if options:
                return self._apply_selection_strategy(options, requested_model)
        
        # 2. 通配符匹配
        for pattern, routes in self.model_routes.items():
            if self._matches_pattern(requested_model, pattern):
                options = self._build_options_from_routes(routes, requested_model)
                if options:
                    return self._apply_selection_strategy(options, requested_model)
        
        # 3. 没有匹配的路由
        return []
    
    def _build_options_from_routes(self, routes: List[ModelRoute], requested_model: str) -> List[Tuple[str, Provider, int]]:
        """从路由配置构建可用选项"""
        options = []
        cooldown = self.get_failure_cooldown()
        
        for route in routes:
            if not route.enabled:
                continue
                
            provider = self.get_provider_by_name_and_account(route.provider, route.account_email)
            if not provider or not provider.enabled or not provider.is_healthy(cooldown):
                continue
            
            # 处理模型名称
            target_model = route.model
            if target_model == "passthrough":
                target_model = requested_model
            
            options.append((target_model, provider, route.priority))
        
        return options
    
    def _apply_selection_strategy(self, options: List[Tuple[str, Provider, int]], requested_model: str) -> List[Tuple[str, Provider]]:
        """根据选择策略对选项进行排序和选择"""
        if not options:
            return []
        
        # 检查粘滞provider是否过期，如果是则使用正常的优先级选择
        current_time = time.time()
        is_sticky_expired = (current_time - self._last_request_time) > self._sticky_provider_duration
        
        if not is_sticky_expired and self._last_successful_provider:
            # 粘滞期间：优先使用上次成功的provider
            sticky_option = None
            other_options = []
            
            for model, provider, priority in options:
                if provider.name == self._last_successful_provider:
                    sticky_option = (model, provider, priority)
                else:
                    other_options.append((model, provider, priority))
            
            # 如果找到了粘滞provider，将其放在第一位
            if sticky_option:
                sorted_other_options = sorted(other_options, key=lambda x: x[2])
                final_options = [sticky_option] + sorted_other_options
                # print(f"[DEBUG] Sticky logic applied - prioritizing {self._last_successful_provider}")
                return [(model, provider) for model, provider, priority in final_options]
        
        if self.selection_strategy == SelectionStrategy.PRIORITY:
            # 按优先级排序（数字越小优先级越高）
            sorted_options = sorted(options, key=lambda x: x[2])
            return [(model, provider) for model, provider, priority in sorted_options]
        
        elif self.selection_strategy == SelectionStrategy.ROUND_ROBIN:
            # Round robin选择
            sorted_options = sorted(options, key=lambda x: x[2])
            key = f"{requested_model}_{len(sorted_options)}"
            
            if key not in self._round_robin_indices:
                self._round_robin_indices[key] = 0
            
            current_index = self._round_robin_indices[key]
            self._round_robin_indices[key] = (current_index + 1) % len(sorted_options)
            
            # 将当前选择的放在第一位，其他保持优先级顺序
            selected = sorted_options.pop(current_index)
            result = [selected] + sorted_options
            return [(model, provider) for model, provider, priority in result]
        
        elif self.selection_strategy == SelectionStrategy.RANDOM:
            # 随机选择，但仍然返回所有选项作为fallback
            sorted_options = sorted(options, key=lambda x: x[2])
            if len(sorted_options) > 1:
                # 从前3个优先级中随机选择
                top_options = sorted_options[:min(3, len(sorted_options))]
                selected = random.choice(top_options)
                remaining = [opt for opt in sorted_options if opt != selected]
                result = [selected] + remaining
                return [(model, provider) for model, provider, priority in result]
            return [(model, provider) for model, provider, priority in sorted_options]
        
        # 默认按优先级排序
        sorted_options = sorted(options, key=lambda x: x[2])
        return [(model, provider) for model, provider, priority in sorted_options]
    
    def get_failure_cooldown(self) -> int:
        """Get failure cooldown time from settings"""
        return self.settings.get('failure_cooldown', 60)
    
    def get_non_streaming_timeouts(self) -> Dict[str, int]:
        """获取非流式请求超时配置"""
        timeouts = self.settings.get('timeouts', {})
        non_streaming = timeouts.get('non_streaming', {})
        return {
            'connect_timeout': non_streaming.get('connect_timeout', 30),
            'read_timeout': non_streaming.get('read_timeout', 60),
            'pool_timeout': non_streaming.get('pool_timeout', 30)
        }
    
    def get_streaming_timeouts(self) -> Dict[str, int]:
        """获取流式请求超时配置"""
        timeouts = self.settings.get('timeouts', {})
        streaming = timeouts.get('streaming', {})
        return {
            'connect_timeout': streaming.get('connect_timeout', 30),
            'read_timeout': streaming.get('read_timeout', 120),
            'pool_timeout': streaming.get('pool_timeout', 30)
        }
    
    def get_timeouts_for_request(self, is_streaming: bool) -> Dict[str, int]:
        """根据请求类型获取相应的超时配置"""
        if is_streaming:
            return self.get_streaming_timeouts()
        else:
            return self.get_non_streaming_timeouts()
    
    def get_caching_timeouts(self) -> Dict[str, int]:
        """获取缓存相关超时配置"""
        timeouts = self.settings.get('timeouts', {})
        caching = timeouts.get('caching', {})
        return {
            'deduplication_timeout': caching.get('deduplication_timeout', 300)
        }
    
    def get_healthy_providers(self) -> List[Provider]:
        """Get list of healthy (non-failed) providers"""
        cooldown = self.get_failure_cooldown()
        
        # 简化逻辑：只返回健康的providers，粘滞逻辑已移至选择策略中
        healthy_providers = [p for p in self.providers if p.enabled and p.is_healthy(cooldown)]
        # Removed debug print - this would be too noisy in production
        return healthy_providers
    
    def mark_provider_success(self, provider_name: str):
        """标记provider成功，更新粘滞状态"""
        self._last_successful_provider = provider_name
        # 在请求成功完成时更新最后请求时间
        self._last_request_time = time.time()
    
    def mark_provider_used(self, provider_name: str):
        """标记provider被使用（无论成功失败），用于sticky逻辑"""
        # 只要没有触发failover，就启用sticky
        self._last_successful_provider = provider_name
        self._last_request_time = time.time()
    
    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        """根据名称获取provider"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
    
    def get_provider_headers(self, provider: Provider, original_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get authentication headers for a provider, optionally merging with original headers"""
        return self.provider_auth.get_provider_headers(provider, original_headers)
    
    def get_request_url(self, provider: Provider, endpoint: str) -> str:
        """Get full request URL for a provider"""
        base_url = provider.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/{endpoint}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers and model routes"""
        status = {
            "total_providers": len(self.providers),
            "healthy_providers": len(self.get_healthy_providers()),
            "selection_strategy": self.selection_strategy.value,
            "total_model_routes": len(self.model_routes),
            "providers": []
        }
        
        cooldown = self.get_failure_cooldown()
        for provider in self.providers:
            provider_status = {
                "name": provider.name,
                "type": provider.type.value,
                "base_url": provider.base_url,
                "enabled": provider.enabled,
                "healthy": provider.is_healthy(cooldown),
                "failure_count": provider.failure_count,
                "last_failure_time": provider.last_failure_time,
                "proxy": provider.proxy
            }
            status["providers"].append(provider_status)
        
        return status
    
    def reload_config(self):
        """Reload configuration from file"""
        self.load_config()

    def handle_oauth_authorization_required(self, provider: Provider, http_status_code: int = 401):
        """Handle 401/403 authorization required error for OAuth providers"""
        return self.provider_auth.handle_oauth_authorization_required(provider, http_status_code)
    
    def get_error_handling_decision(self, error: Exception, http_status_code: Optional[int] = None, is_streaming: bool = False) -> tuple[str, bool, bool]:
        """
        获取错误处理决策 - 判断是否标记不健康和是否可以故障转移
        
        Args:
            error: 捕获的异常
            http_status_code: HTTP状态码（如果有）
            is_streaming: 是否为streaming请求
            
        Returns:
            tuple: (error_reason, should_mark_unhealthy, can_failover)
        """
        should_mark_unhealthy, can_failover, error_reason = get_error_handling_decision(
            error, http_status_code, is_streaming,
            self.settings.get('unhealthy_http_codes', []),
            self.settings.get('unhealthy_exception_patterns', [])
        )
        return error_reason, should_mark_unhealthy, can_failover

    def update_provider_auth(self, provider_name: str, new_auth_value: str):
        """更新provider的认证值（用于token刷新）"""
        provider = self.get_provider_by_name(provider_name)
        if provider:
            provider.auth_value = new_auth_value
            return True
        return False

    def reset_all_provider_states(self):
        """重置所有provider的状态（用于测试）"""
        for provider in self.providers:
            provider.mark_success()  # Reset failure count and last_failure_time
    
    
    def check_and_reset_timeout_errors(self, request_id: str = ""):
        """检查并重置超时的错误计数（内部辅助函数）"""
        if self.unhealthy_reset_timeout <= 0:
            return  # 如果timeout配置为0或负数，跳过timeout reset
        
        current_time = time.time()
        providers_to_reset = []
        
        # 查找需要重置的providers
        for provider in self.providers:
            if (provider.failure_count > 0 and 
                provider.last_unhealthy_time and 
                current_time - provider.last_unhealthy_time > self.unhealthy_reset_timeout):
                providers_to_reset.append(provider)
        
        # 执行重置
        for provider in providers_to_reset:
            old_count = provider.failure_count
            provider.mark_success()  # 重置所有状态
            
            # 记录日志
            debug(LogRecord(
                LogEvent.PROVIDER_HEALTH_ERROR_COUNT_TIMEOUT_RESET.value,
                f"Provider {provider.name} error count reset from {old_count} to 0 after timeout ({self.unhealthy_reset_timeout}s)",
                request_id,
                {
                    "provider": provider.name,
                    "old_error_count": old_count,
                    "reset_reason": "timeout",
                    "timeout_seconds": self.unhealthy_reset_timeout
                }
            ))
    
    def record_health_check_result(self, provider_name: str, is_error_detected: bool, error_type: Optional[str] = None, request_id: str = "") -> bool:
        """记录健康检查结果，返回是否应该标记为unhealthy"""
        provider = self.get_provider_by_name(provider_name)
        if not provider:
            return False
            
        if is_error_detected:
            provider.mark_failure()
            
            # 记录错误详细信息
            debug(LogRecord(
                LogEvent.PROVIDER_HEALTH_ERROR_RECORDED.value,
                f"Recorded error for provider {provider_name}: count={provider.failure_count}/{self.unhealthy_threshold}, reason={error_type or 'unknown'}",
                request_id,
                {
                    "provider": provider_name,
                    "error_count": provider.failure_count,
                    "threshold": self.unhealthy_threshold,
                    "error_reason": error_type or "unknown"
                }
            ))
            
            # Check if provider should be marked unhealthy based on threshold
            should_mark_unhealthy = provider.failure_count >= self.unhealthy_threshold
            
            if should_mark_unhealthy:
                # 标记为unhealthy时更新last_unhealthy_time
                provider.last_unhealthy_time = time.time()
                
                warning(LogRecord(
                    LogEvent.PROVIDER_MARKED_UNHEALTHY.value,
                    f"Provider {provider_name} marked unhealthy after {provider.failure_count} errors (threshold: {self.unhealthy_threshold})",
                    request_id,
                    {
                        "provider": provider_name,
                        "error_count": provider.failure_count,
                        "threshold": self.unhealthy_threshold,
                        "error_reason": error_type or "unknown"
                    }
                ))
            else:
                # 错误数不够，不标记unhealthy，但需要记录状态
                pass
            
            return should_mark_unhealthy
        else:
            # Success case - reset failures if enabled
            if self.unhealthy_reset_on_success and provider.failure_count > 0:
                old_count = provider.failure_count
                provider.mark_success()
                
                debug(LogRecord(
                    LogEvent.PROVIDER_HEALTH_ERROR_COUNT_RESET.value,
                    f"Provider {provider_name} error count reset from {old_count} to 0 after success",
                    request_id,
                    {
                        "provider": provider_name,
                        "old_error_count": old_count,
                        "reset_reason": "success"
                    }
                ))
            
            return False
    
    def get_provider_error_status(self, provider_name: str) -> Dict[str, Any]:
        """获取Provider的错误状态信息"""
        provider = self.get_provider_by_name(provider_name)
        if not provider:
            return {
                "provider": provider_name,
                "error": "Provider not found"
            }
            
        return {
            "error_count": provider.failure_count,
            "threshold": self.unhealthy_threshold,
            "last_error_time": provider.last_failure_time,
            "last_success_time": provider.last_success_time,
            "reset_on_success": self.unhealthy_reset_on_success,
            "reset_timeout": self.unhealthy_reset_timeout
        }

    def shutdown(self):
        """Shutdown the provider manager"""
        pass