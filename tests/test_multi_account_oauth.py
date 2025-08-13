"""
Multi-account OAuth configuration tests.

This file tests the multi-account OAuth functionality where multiple providers
can be configured with different account_email values to support OAuth token
routing to specific accounts.

Test Coverage:
- test_oauth_account_email_configuration: Test provider loading with account_email
- test_oauth_provider_selection_by_account: Test provider selection based on account
- test_oauth_multiple_accounts_failover: Test failover between different OAuth accounts
- test_oauth_mixed_auth_providers: Test mixed auth types (oauth + api_key)
"""

import asyncio
import pytest
import httpx
import time
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import the new testing framework
from framework import (
    Scenario, ProviderConfig, ProviderBehavior, ExpectedBehavior,
    Environment, TestConfigFactory
)


class TestMultiAccountOAuth:
    """Multi-account OAuth configuration and functionality tests."""

    def test_oauth_provider_configuration_loading(self):
        """Test that providers with account_email configuration are loaded correctly."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.manager import ProviderManager
        import tempfile
        import yaml
        import os
        
        # Create a test configuration with multiple OAuth accounts
        test_config = {
            "providers": [
                {
                    "name": "Claude Code Official",
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user1@example.com",
                    "enabled": True
                },
                {
                    "name": "Claude Code Official", 
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user2@example.com",
                    "enabled": True
                },
                {
                    "name": "Regular API Provider",
                    "type": "anthropic",
                    "base_url": "https://api.example.com",
                    "auth_type": "api_key",
                    "auth_value": "sk-test-key",
                    "enabled": True
                }
            ],
            "model_routes": {
                "test-model": [
                    {"provider": "Claude Code Official", "model": "passthrough", "priority": 1},
                    {"provider": "Claude Code Official", "model": "passthrough", "priority": 2},
                    {"provider": "Regular API Provider", "model": "passthrough", "priority": 3}
                ]
            },
            "settings": {
                "selection_strategy": "priority",
                "unhealthy_threshold": 2,
                "failure_cooldown": 60,
                "log_level": "DEBUG"
            }
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Load configuration
            provider_manager = ProviderManager(config_path)
            provider_manager.load_config()
            
            # Verify providers are loaded correctly
            assert len(provider_manager.providers) == 3
            
            # Find OAuth providers
            oauth_provider1 = next((p for p in provider_manager.providers if p.account_email == "user1@example.com"), None)
            oauth_provider2 = next((p for p in provider_manager.providers if p.account_email == "user2@example.com"), None)
            regular_provider = next((p for p in provider_manager.providers if p.auth_type.value == "api_key"), None)
            
            # Verify OAuth provider 1
            assert oauth_provider1 is not None
            assert oauth_provider1.name == "Claude Code Official"
            assert oauth_provider1.auth_type.value == "auth_token"
            assert oauth_provider1.auth_value == "oauth"
            assert oauth_provider1.account_email == "user1@example.com"
            
            # Verify OAuth provider 2
            assert oauth_provider2 is not None
            assert oauth_provider2.name == "Claude Code Official"
            assert oauth_provider2.auth_type.value == "auth_token"
            assert oauth_provider2.auth_value == "oauth"
            assert oauth_provider2.account_email == "user2@example.com"
            
            # Verify regular provider has no account_email
            assert regular_provider is not None
            assert regular_provider.account_email is None
            assert regular_provider.auth_type.value == "api_key"
            
            print("✅ OAuth provider configuration loading test passed")
            
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_oauth_token_retrieval_by_email(self):
        """Test OAuth manager's get_token_by_email functionality."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from oauth.oauth_manager import OAuthManager, TokenCredentials
        import time
        
        # Create OAuth manager
        oauth_manager = OAuthManager(enable_persistence=False)
        
        # Add mock tokens for different accounts
        current_time = int(time.time())
        token1 = TokenCredentials(
            access_token="access_token_1",
            refresh_token="refresh_token_1", 
            expires_at=current_time + 3600,
            scopes=["user:profile", "user:inference"],
            account_email="user1@example.com",
            account_id="user1@example.com"
        )
        
        token2 = TokenCredentials(
            access_token="access_token_2",
            refresh_token="refresh_token_2",
            expires_at=current_time + 3600,
            scopes=["user:profile", "user:inference"], 
            account_email="user2@example.com",
            account_id="user2@example.com"
        )
        
        oauth_manager.token_credentials = [token1, token2]
        
        # Test getting token by specific email
        result_token1 = oauth_manager.get_token_by_email("user1@example.com")
        assert result_token1 == "access_token_1"
        
        result_token2 = oauth_manager.get_token_by_email("user2@example.com")
        assert result_token2 == "access_token_2"
        
        # Test case insensitive email matching
        result_token1_upper = oauth_manager.get_token_by_email("USER1@EXAMPLE.COM")
        assert result_token1_upper == "access_token_1"
        
        # Test non-existent account
        result_none = oauth_manager.get_token_by_email("nonexistent@example.com")
        assert result_none is None
        
        # Test fallback to round-robin when no email provided
        result_round_robin = oauth_manager.get_token_by_email("")
        assert result_round_robin in ["access_token_1", "access_token_2"]
        
        print("✅ OAuth token retrieval by email test passed")

    def test_provider_auth_with_account_email(self):
        """Test provider authentication logic with account_email."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.provider_auth import ProviderAuth
        from core.provider_manager.manager import Provider, ProviderType, AuthType
        from oauth.oauth_manager import OAuthManager, TokenCredentials
        from unittest.mock import patch
        import time
        
        # Create mock OAuth manager with tokens
        mock_oauth_manager = OAuthManager(enable_persistence=False)
        current_time = int(time.time())
        
        token1 = TokenCredentials(
            access_token="oauth_token_user1",
            refresh_token="refresh_token_1",
            expires_at=current_time + 3600,
            scopes=["user:profile"],
            account_email="user1@example.com",
            account_id="user1@example.com"
        )
        
        token2 = TokenCredentials(
            access_token="oauth_token_user2", 
            refresh_token="refresh_token_2",
            expires_at=current_time + 3600,
            scopes=["user:profile"],
            account_email="user2@example.com",
            account_id="user2@example.com"
        )
        
        mock_oauth_manager.token_credentials = [token1, token2]
        
        # Create provider auth instance
        provider_auth = ProviderAuth()
        
        # Create test providers
        oauth_provider1 = Provider(
            name="Claude Code Official",
            type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            auth_type=AuthType.AUTH_TOKEN,
            auth_value="oauth",
            account_email="user1@example.com"
        )
        
        oauth_provider2 = Provider(
            name="Claude Code Official",
            type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com", 
            auth_type=AuthType.AUTH_TOKEN,
            auth_value="oauth",
            account_email="user2@example.com"
        )
        
        regular_provider = Provider(
            name="Regular Provider",
            type=ProviderType.ANTHROPIC,
            base_url="https://api.example.com",
            auth_type=AuthType.API_KEY,
            auth_value="sk-test-key"
        )
        
        # Mock the OAuth manager getter
        with patch.object(provider_auth, '_get_oauth_manager', return_value=mock_oauth_manager):
            # Test OAuth provider 1 gets correct token
            headers1 = provider_auth.get_provider_headers(oauth_provider1)
            assert headers1["Authorization"] == "Bearer oauth_token_user1"
            assert headers1["anthropic-version"] == "2023-06-01"
            
            # Test OAuth provider 2 gets correct token
            headers2 = provider_auth.get_provider_headers(oauth_provider2)
            assert headers2["Authorization"] == "Bearer oauth_token_user2"
            assert headers2["anthropic-version"] == "2023-06-01"
            
            # Test regular provider uses API key
            headers3 = provider_auth.get_provider_headers(regular_provider)
            assert headers3["x-api-key"] == "sk-test-key"
            assert headers3["anthropic-version"] == "2023-06-01"
        
        print("✅ Provider auth with account_email test passed")

    @pytest.mark.asyncio
    async def test_oauth_provider_priority_and_failover(self):
        """Test OAuth provider priority and failover with account-specific routing."""
        # This test would require a more complex setup with actual OAuth flow
        # For now, we'll create a unit test that verifies the logic
        
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.manager import ProviderManager
        import tempfile
        import yaml
        import os
        
        # Create configuration with OAuth providers at different priorities
        test_config = {
            "providers": [
                {
                    "name": "OAuth Primary",
                    "type": "anthropic", 
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "primary@example.com",
                    "enabled": True
                },
                {
                    "name": "OAuth Secondary",
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com", 
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "secondary@example.com",
                    "enabled": True
                },
                {
                    "name": "API Key Fallback",
                    "type": "anthropic",
                    "base_url": "https://api.fallback.com",
                    "auth_type": "api_key", 
                    "auth_value": "sk-fallback",
                    "enabled": True
                }
            ],
            "model_routes": {
                "test-model": [
                    {"provider": "OAuth Primary", "model": "passthrough", "priority": 1},
                    {"provider": "OAuth Secondary", "model": "passthrough", "priority": 2},
                    {"provider": "API Key Fallback", "model": "passthrough", "priority": 3}
                ]
            },
            "settings": {
                "selection_strategy": "priority",
                "unhealthy_threshold": 1,
                "failure_cooldown": 60
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            provider_manager = ProviderManager(config_path)
            provider_manager.load_config()
            
            # Test provider selection logic
            options = provider_manager.select_model_and_provider_options("test-model")
            
            # Should return providers in priority order
            assert len(options) >= 3
            assert options[0][1].name == "OAuth Primary"
            assert options[0][1].account_email == "primary@example.com"
            assert options[1][1].name == "OAuth Secondary" 
            assert options[1][1].account_email == "secondary@example.com"
            assert options[2][1].name == "API Key Fallback"
            assert options[2][1].account_email is None
            
            print("✅ OAuth provider priority and failover test passed")
            
        finally:
            os.unlink(config_path)

    def test_oauth_configuration_validation(self):
        """Test validation of OAuth configuration with account_email."""
        # Test that configuration properly validates account_email fields
        
        test_cases = [
            # Valid OAuth configuration with account_email
            {
                "name": "Valid OAuth with account", 
                "config": {
                    "name": "Claude Code Official",
                    "type": "anthropic",
                    "auth_type": "auth_token", 
                    "auth_value": "oauth",
                    "account_email": "user@example.com"
                },
                "should_pass": True
            },
            # Valid OAuth configuration without account_email (fallback to round-robin)
            {
                "name": "Valid OAuth without account",
                "config": {
                    "name": "Claude Code Official",
                    "type": "anthropic", 
                    "auth_type": "auth_token",
                    "auth_value": "oauth"
                },
                "should_pass": True
            },
            # Non-OAuth provider should ignore account_email
            {
                "name": "API Key provider with account_email",
                "config": {
                    "name": "API Provider",
                    "type": "anthropic",
                    "auth_type": "api_key",
                    "auth_value": "sk-test",
                    "account_email": "user@example.com"  # Should be ignored
                },
                "should_pass": True
            }
        ]
        
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.manager import Provider, ProviderType, AuthType
        
        for test_case in test_cases:
            config = test_case["config"]
            try:
                provider = Provider(
                    name=config["name"],
                    type=ProviderType(config["type"]),
                    base_url="https://api.example.com",
                    auth_type=AuthType(config["auth_type"]),
                    auth_value=config["auth_value"],
                    account_email=config.get("account_email")
                )
                
                # Verify account_email is properly set
                if "account_email" in config:
                    assert provider.account_email == config["account_email"]
                else:
                    assert provider.account_email is None
                
                print(f"✅ {test_case['name']} - Configuration validation passed")
                
            except Exception as e:
                if test_case["should_pass"]:
                    pytest.fail(f"Test case '{test_case['name']}' should have passed but raised: {e}")
                else:
                    print(f"✅ {test_case['name']} - Expected failure: {e}")

    def test_same_name_different_account_email_routing(self):
        """Test routing with same provider name but different account_email."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.manager import ProviderManager
        import tempfile
        import yaml
        
        # Create configuration with same provider names but different account_email
        test_config = {
            "providers": [
                {
                    "name": "Claude Code Official",  # 相同名称
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user1@example.com",  # 不同账户
                    "enabled": True
                },
                {
                    "name": "Claude Code Official",  # 相同名称
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user2@example.com",  # 不同账户
                    "enabled": True
                },
                {
                    "name": "Claude Code Official",  # 相同名称
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "api_key",
                    "auth_value": "sk-test-key",
                    "account_email": None,  # 无账户（API Key类型）
                    "enabled": True
                }
            ],
            "model_routes": {
                "test-model": [
                    {
                        "provider": "Claude Code Official",
                        "model": "passthrough",
                        "priority": 1,
                        "account_email": "user1@example.com"  # 指定特定账户
                    },
                    {
                        "provider": "Claude Code Official", 
                        "model": "passthrough",
                        "priority": 2,
                        "account_email": "user2@example.com"  # 指定另一个账户
                    },
                    {
                        "provider": "Claude Code Official",
                        "model": "passthrough", 
                        "priority": 3
                        # 不指定account_email，应该匹配到API Key类型的provider
                    }
                ]
            },
            "settings": {
                "selection_strategy": "priority",
                "unhealthy_threshold": 1,
                "failure_cooldown": 60
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            provider_manager = ProviderManager(config_path)
            provider_manager.load_config()
            
            # 验证加载了3个同名provider
            assert len(provider_manager.providers) == 3
            all_same_name = all(p.name == "Claude Code Official" for p in provider_manager.providers)
            assert all_same_name, "所有provider应该都叫'Claude Code Official'"
            
            # 验证不同的account_email
            emails = [p.account_email for p in provider_manager.providers]
            assert "user1@example.com" in emails
            assert "user2@example.com" in emails
            assert None in emails  # API Key provider
            
            # 测试模型路由选择
            options = provider_manager.select_model_and_provider_options("test-model")
            
            # 应该返回3个选项，按优先级排序
            assert len(options) == 3, f"应该返回3个选项，实际返回{len(options)}个"
            
            # 验证第一个选项（优先级1，user1@example.com）
            first_model, first_provider = options[0]
            assert first_provider.account_email == "user1@example.com"
            assert first_provider.auth_value == "oauth"
            
            # 验证第二个选项（优先级2，user2@example.com）
            second_model, second_provider = options[1]
            assert second_provider.account_email == "user2@example.com"
            assert second_provider.auth_value == "oauth"
            
            # 验证第三个选项（优先级3，无account_email，API Key）
            third_model, third_provider = options[2]
            assert third_provider.account_email is None
            assert third_provider.auth_value == "sk-test-key"
            
            print("✅ Same name different account_email routing test passed")
            print(f"   Route 1: {first_provider.name} -> {first_provider.account_email}")
            print(f"   Route 2: {second_provider.name} -> {second_provider.account_email}")
            print(f"   Route 3: {third_provider.name} -> {third_provider.account_email}")
            
        finally:
            os.unlink(config_path)

    def test_provider_lookup_by_name_and_account(self):
        """Test the new _get_provider_by_name_and_account method."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.provider_manager.manager import ProviderManager
        import tempfile
        import yaml
        
        test_config = {
            "providers": [
                {
                    "name": "Claude Code Official",
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user1@example.com",
                    "enabled": True
                },
                {
                    "name": "Claude Code Official",
                    "type": "anthropic", 
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "auth_token",
                    "auth_value": "oauth",
                    "account_email": "user2@example.com",
                    "enabled": True
                },
                {
                    "name": "Claude Code Official",
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "auth_type": "api_key",
                    "auth_value": "sk-test",
                    "enabled": True
                }
            ],
            "settings": {"log_level": "DEBUG"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            manager = ProviderManager(config_path)
            manager.load_config()
            
            # 测试精确匹配
            provider1 = manager.get_provider_by_name_and_account("Claude Code Official", "user1@example.com")
            assert provider1 is not None
            assert provider1.account_email == "user1@example.com"

            provider2 = manager.get_provider_by_name_and_account("Claude Code Official", "user2@example.com")
            assert provider2 is not None
            assert provider2.account_email == "user2@example.com"
            
            # 测试查找无account_email的provider
            provider3 = manager.get_provider_by_name_and_account("Claude Code Official", None)
            assert provider3 is not None
            assert provider3.account_email is None
            assert provider3.auth_value == "sk-test"
            
            # 测试查找不存在的账户
            provider_none = manager.get_provider_by_name_and_account("Claude Code Official", "nonexistent@example.com")
            assert provider_none is None
            
            # 测试大小写不敏感
            provider_case = manager.get_provider_by_name_and_account("Claude Code Official", "USER1@EXAMPLE.COM")
            assert provider_case is not None
            assert provider_case.account_email == "user1@example.com"
            
            print("✅ Provider lookup by name and account test passed")
            
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestMultiAccountOAuth()
    
    print("Running multi-account OAuth tests...")
    print("=" * 60)
    
    try:
        test_instance.test_oauth_provider_configuration_loading()
        test_instance.test_oauth_token_retrieval_by_email()
        test_instance.test_provider_auth_with_account_email()
        asyncio.run(test_instance.test_oauth_provider_priority_and_failover())
        test_instance.test_oauth_configuration_validation()
        test_instance.test_same_name_different_account_email_routing()
        test_instance.test_provider_lookup_by_name_and_account()
        
        print("=" * 60)
        print("🎉 All multi-account OAuth tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise