#!/usr/bin/env python3
"""
测试脚本：验证增强版provider管理器和智能路由功能
"""

import asyncio
import sys
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.intelligent_router import SimpleRouter, RouteDecision
from src.provider_types import ProviderConfig, ProviderType, AuthType, BillingModel


def test_routing_decisions():
    """测试智能路由决策"""
    print("🧠 测试智能路由决策...")
    
    router = SimpleRouter()
    
    # 测试用例
    test_cases = [
        {
            "name": "简单问答",
            "request": {
                "messages": [{"role": "user", "content": "什么是Python？"}],
                "tools": []
            },
            "expected": RouteDecision.TOKEN_BASED
        },
        {
            "name": "多工具任务",
            "request": {
                "messages": [{"role": "user", "content": "分析项目中的所有Python文件，找出性能问题并生成报告"}],
                "tools": ["read_file", "search_code", "analyze_performance", "generate_report"]
            },
            "expected": RouteDecision.SESSION_BASED
        },
        {
            "name": "长文本分析",
            "request": {
                "messages": [{"role": "user", "content": "请详细分析这个项目的架构设计，包括：" + "模块设计、数据流、错误处理、性能优化、扩展性考虑、安全措施、部署策略、监控方案、测试覆盖、文档完整性。" * 20}],
                "tools": ["read_file", "search_code"]
            },
            "expected": RouteDecision.SESSION_BASED
        },
        {
            "name": "代码生成",
            "request": {
                "messages": [{"role": "user", "content": "写一个Python函数来计算斐波那契数列"}],
                "tools": []
            },
            "expected": RouteDecision.TOKEN_BASED
        },
        {
            "name": "文件处理任务",
            "request": {
                "messages": [{"role": "user", "content": "搜索项目中的config.yaml、settings.json和.env文件，分析配置项"}],
                "tools": ["search_files", "read_file"]
            },
            "expected": RouteDecision.SESSION_BASED
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试案例 {i}: {test_case['name']}")
        
        decision, details = router.route_request(test_case["request"])
        
        # 显示结果
        print(f"   决策: {decision.value}")
        print(f"   原因: {details['reason']}")
        print(f"   置信度: {details['confidence']:.2f}")
        
        # 显示特征
        features = details["features"]
        print(f"   特征: 工具{features['tool_count']}个, 文本{features['text_length']}字符")
        
        # 验证结果
        if decision == test_case["expected"]:
            print("   ✅ 决策正确")
        else:
            print(f"   ❌ 决策错误，期望{test_case['expected'].value}")
    
    print(f"\n✅ 路由决策测试完成")


def test_provider_types():
    """测试Provider类型系统"""
    print("\n🏗️  测试Provider类型系统...")
    
    # 创建不同类型的providers
    token_provider = ProviderConfig(
        name="test_token",
        type=ProviderType.ANTHROPIC,
        base_url="https://api.anthropic.com",
        auth_type=AuthType.API_KEY,
        auth_value="sk-test"
    )
    
    session_provider = ProviderConfig(
        name="test_session",
        type=ProviderType.ZED,
        base_url="https://zed-api.example.com",
        auth_type=AuthType.API_KEY,
        auth_value="zed-test"
    )
    
    # 测试自动推断
    print(f"Token Provider - 计费模式: {token_provider.get_billing_model().value}")
    print(f"Session Provider - 计费模式: {session_provider.get_billing_model().value}")
    
    # 测试判断方法
    print(f"Token Provider - is_token_based: {token_provider.is_token_based()}")
    print(f"Session Provider - is_session_based: {session_provider.is_session_based()}")
    
    print("✅ Provider类型测试完成")


def test_configuration_loading():
    """测试配置加载"""
    print("\n⚙️  测试配置文件格式...")
    
    try:
        import yaml
        
        # 读取enhanced配置文件
        config_path = Path(__file__).parent / "providers.enhanced.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("📄 配置文件结构:")
            print(f"   Providers: {len(config.get('providers', []))}")
            print(f"   路由启用: {config.get('routing', {}).get('enabled', False)}")
            print(f"   模型路由: {len(config.get('model_routes', {}))}")
            
            # 分析providers
            billing_models = {}
            for provider in config.get('providers', []):
                billing = provider.get('billing_model', 'auto')
                billing_models[billing] = billing_models.get(billing, 0) + 1
            
            print(f"   计费模式分布: {billing_models}")
            
            print("✅ 配置文件格式正确")
        else:
            print("❌ 配置文件不存在")
    
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")


def display_routing_examples():
    """显示路由示例"""
    print("\n📖 智能路由示例:")
    
    examples = [
        {
            "scenario": "简单问答",
            "request": "什么是机器学习？",
            "tools": 0,
            "expected_route": "Token-based",
            "reason": "简单问答，成本低"
        },
        {
            "scenario": "代码分析",
            "request": "分析这个Python项目的架构",
            "tools": 3,
            "expected_route": "Session-based", 
            "reason": "多工具调用，固定成本更优"
        },
        {
            "scenario": "文档处理",
            "request": "搜索所有markdown文件并生成目录",
            "tools": 2,
            "expected_route": "Session-based",
            "reason": "涉及搜索和文件操作"
        },
        {
            "scenario": "快速生成",
            "request": "写一个hello world函数",
            "tools": 0,
            "expected_route": "Token-based",
            "reason": "简单生成任务"
        }
    ]
    
    for example in examples:
        print(f"\n🔍 场景: {example['scenario']}")
        print(f"   请求: {example['request']}")
        print(f"   工具数: {example['tools']}")
        print(f"   推荐路由: {example['expected_route']}")
        print(f"   理由: {example['reason']}")


def main():
    """主函数"""
    print("🚀 Enhanced Provider Manager 测试")
    print("=" * 50)
    
    # 运行各项测试
    test_provider_types()
    test_routing_decisions()
    test_configuration_loading()
    display_routing_examples()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成！")
    
    print("\n💡 使用建议:")
    print("1. 启用智能路由可以自动选择最优计费模式")
    print("2. Session-based适合复杂任务，Token-based适合简单任务")
    print("3. 配置关键词权重可以优化路由决策")
    print("4. 监控路由决策日志来调优策略")


if __name__ == "__main__":
    main()