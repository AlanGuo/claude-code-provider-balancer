"""
智能路由器
根据请求特征选择最优的计费模式和provider
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import re
from .provider_types import BillingModel, ProviderConfig


class RouteDecision(str, Enum):
    """路由决策"""
    TOKEN_BASED = "token_based"
    SESSION_BASED = "session_based"


class SimpleRouter:
    """简化的智能路由器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化路由器
        
        Args:
            config: 路由配置，如果为None则使用默认配置
        """
        self.config = config or self._get_default_config()
        
        # 提取关键词配置
        self.session_keywords = self.config.get('session_keywords', {})
        self.token_keywords = self.config.get('token_keywords', {})
        
        # 提取强制规则配置
        self.force_session_rules = self.config.get('force_session_based', [])
        self.force_token_rules = self.config.get('force_token_based', [])
        
        # 默认策略
        self.default_with_tools = self.config.get('default_with_tools', 'session_based')
        self.default_without_tools = self.config.get('default_without_tools', 'token_based')
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'force_session_based': [
                {'tool_count_gte': 3},
                {'text_length_gte': 2000},
                {'has_multiple_files': True}
            ],
            'force_token_based': [
                {'tool_count': 0, 'text_length_lte': 200, 'simple_question': True}
            ],
            'session_keywords': {
                '搜索': 2, '分析': 2, '调试': 2, '扫描': 2,
                '项目': 1, '步骤': 1, '继续': 1, '遍历': 1,
                '优化': 1, '重构': 1, '流程': 1, '计划': 1
            },
            'token_keywords': {
                '什么是': 2, '如何': 2, '解释': 2, '为什么': 2,
                '写一个': 1, '创建一个': 1, '定义': 1, '介绍': 1
            },
            'default_with_tools': 'session_based',
            'default_without_tools': 'token_based'
        }
    
    def route_request(self, request: Dict[str, Any]) -> Tuple[RouteDecision, Dict[str, Any]]:
        """
        路由请求到最优的计费模式
        
        Args:
            request: 原始请求
            
        Returns:
            (决策结果, 决策详情)
        """
        # 提取请求特征
        features = self._extract_features(request)
        
        # 检查强制规则
        force_decision = self._check_force_rules(features)
        if force_decision:
            return force_decision[0], {
                'decision': force_decision[0].value,
                'reason': f'强制规则: {force_decision[1]}',
                'features': features,
                'confidence': 1.0
            }
        
        # 关键词匹配评分
        session_score = self._calculate_keyword_score(features['text_content'], self.session_keywords)
        token_score = self._calculate_keyword_score(features['text_content'], self.token_keywords)
        
        # 基于评分决策
        if session_score > token_score + 1:
            decision = RouteDecision.SESSION_BASED
            reason = f'Session关键词评分({session_score}) > Token关键词评分({token_score})'
        elif token_score > session_score + 1:
            decision = RouteDecision.TOKEN_BASED
            reason = f'Token关键词评分({token_score}) > Session关键词评分({session_score})'
        else:
            # 使用默认策略
            if features['tool_count'] > 0:
                decision = RouteDecision.SESSION_BASED if self.default_with_tools == 'session_based' else RouteDecision.TOKEN_BASED
                reason = f'默认策略(有工具): {self.default_with_tools}'
            else:
                decision = RouteDecision.TOKEN_BASED if self.default_without_tools == 'token_based' else RouteDecision.SESSION_BASED
                reason = f'默认策略(无工具): {self.default_without_tools}'
        
        # 计算置信度
        confidence = self._calculate_confidence(features, session_score, token_score)
        
        return decision, {
            'decision': decision.value,
            'reason': reason,
            'features': features,
            'scores': {
                'session_keywords': session_score,
                'token_keywords': token_score
            },
            'confidence': confidence
        }
    
    def _extract_features(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """提取请求特征"""
        tools = request.get('tools', [])
        messages = request.get('messages', [])
        
        # 提取文本内容
        text_content = self._extract_text_content(messages)
        
        return {
            'tool_count': len(tools),
            'text_content': text_content,
            'text_length': len(text_content),
            'has_multiple_files': self._has_multiple_file_paths(text_content),
            'is_simple_question': self._is_simple_question(text_content),
            'message_count': len(messages)
        }
    
    def _check_force_rules(self, features: Dict[str, Any]) -> Optional[Tuple[RouteDecision, str]]:
        """检查强制规则"""
        # 检查强制使用session-based的规则
        for rule in self.force_session_rules:
            if self._match_rule(rule, features):
                return RouteDecision.SESSION_BASED, str(rule)
        
        # 检查强制使用token-based的规则
        for rule in self.force_token_rules:
            if self._match_rule(rule, features):
                return RouteDecision.TOKEN_BASED, str(rule)
        
        return None
    
    def _match_rule(self, rule: Dict[str, Any], features: Dict[str, Any]) -> bool:
        """匹配规则"""
        for key, value in rule.items():
            if key == 'tool_count_gte':
                if features['tool_count'] < value:
                    return False
            elif key == 'tool_count':
                if features['tool_count'] != value:
                    return False
            elif key == 'text_length_gte':
                if features['text_length'] < value:
                    return False
            elif key == 'text_length_lte':
                if features['text_length'] > value:
                    return False
            elif key == 'has_multiple_files':
                if features['has_multiple_files'] != value:
                    return False
            elif key == 'simple_question':
                if features['is_simple_question'] != value:
                    return False
        
        return True
    
    def _calculate_keyword_score(self, text: str, keywords: Dict[str, int]) -> int:
        """计算关键词匹配评分"""
        text_lower = text.lower()
        score = 0
        
        for keyword, weight in keywords.items():
            if keyword.lower() in text_lower:
                score += weight
        
        return score
    
    def _extract_text_content(self, messages: List[Dict]) -> str:
        """提取消息中的文本内容"""
        texts = []
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        texts.append(part.get('text', ''))
        return ' '.join(texts)
    
    def _has_multiple_file_paths(self, text: str) -> bool:
        """检测是否包含多个文件路径"""
        # 匹配文件路径模式
        path_patterns = [
            r'[./][\w/\-\.]+\.(py|js|ts|java|cpp|c|h|md|txt|yaml|json|yml)',  # 文件路径
            r'src/[\w/\-\.]+',  # src目录
            r'[\w\-\.]+/[\w\-\.]+/',  # 目录结构
        ]
        
        path_count = 0
        for pattern in path_patterns:
            path_count += len(re.findall(pattern, text))
        
        return path_count >= 2
    
    def _is_simple_question(self, text: str) -> bool:
        """检测是否为简单问答"""
        return len(text) <= 200 and ('?' in text or '？' in text)
    
    def _calculate_confidence(self, features: Dict[str, Any], session_score: int, token_score: int) -> float:
        """计算决策置信度"""
        confidence = 0.7  # 基础置信度
        
        # 明确的工具调用增加置信度
        if features['tool_count'] > 0:
            confidence += 0.1
        
        # 评分差距大增加置信度
        score_diff = abs(session_score - token_score)
        if score_diff >= 3:
            confidence += 0.2
        elif score_diff >= 2:
            confidence += 0.1
        
        # 极端情况增加置信度
        if features['tool_count'] >= 3 or features['text_length'] >= 2000:
            confidence += 0.1
        
        return min(1.0, confidence)


class IntelligentProviderSelector:
    """智能Provider选择器"""
    
    def __init__(self, router: SimpleRouter):
        self.router = router
    
    def select_providers(self, request: Dict[str, Any], available_providers: List[ProviderConfig]) -> Tuple[List[ProviderConfig], Dict[str, Any]]:
        """
        选择最适合的providers
        
        Args:
            request: 原始请求
            available_providers: 可用的providers列表
            
        Returns:
            (推荐的providers列表, 路由详情)
        """
        # 获取路由决策
        decision, route_info = self.router.route_request(request)
        
        # 根据决策过滤providers
        if decision == RouteDecision.SESSION_BASED:
            preferred_providers = [p for p in available_providers if p.is_session_based()]
            fallback_providers = [p for p in available_providers if p.is_token_based()]
        else:
            preferred_providers = [p for p in available_providers if p.is_token_based()]
            fallback_providers = [p for p in available_providers if p.is_session_based()]
        
        # 合并列表，preferred在前
        final_providers = preferred_providers + fallback_providers
        
        # 更新路由信息
        route_info.update({
            'preferred_count': len(preferred_providers),
            'fallback_count': len(fallback_providers),
            'total_providers': len(final_providers)
        })
        
        return final_providers, route_info


def create_router_from_config(config: Optional[Dict[str, Any]] = None) -> SimpleRouter:
    """从配置创建路由器"""
    return SimpleRouter(config)


def route_request_simple(request: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """简单的路由函数"""
    router = SimpleRouter(config)
    decision, details = router.route_request(request)
    
    return {
        'decision': decision.value,
        'details': details
    }