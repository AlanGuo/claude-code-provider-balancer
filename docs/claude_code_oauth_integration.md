# Claude Code OAuth 集成与多账户管理完整指南

## 概述

Claude Code Provider Balancer 提供完整的 OAuth 2.0 认证集成，支持多账户管理和智能路由。本指南涵盖：

- 🔐 自动OAuth授权流程
- 👥 多账户配置和管理  
- 🔄 智能轮换机制
- 💾 安全持久化存储
- ⚡ 自动故障转移
- 🕐 自动token刷新

## 核心功能特性

### 1. OAuth 2.0 集成
- OAuth 2.0 PKCE 安全流程
- 状态参数防CSRF攻击
- 最小权限原则
- 自动token刷新（过期前5分钟）

### 2. 多账户支持
- 相同Provider Name支持不同OAuth账户
- 账户特定的模型路由
- 独立的token管理和刷新
- 使用统计和负载均衡

### 3. 安全存储
- 系统keyring持久化存储
- 重启后自动加载token
- 支持多用户环境
- 敏感数据加密保护

## 配置指南

### 基础OAuth配置

```yaml
providers:
  # 单个OAuth账户配置
  - name: "Claude Code Official"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    auth_type: "auth_token"
    auth_value: "oauth"  # 使用OAuth token认证
    enabled: true

# 全局OAuth设置
settings:
  oauth:
    enable_auto_refresh: true  # 启用自动刷新
    enable_persistence: true   # 启用keyring持久化存储
    service_name: "claude-code-balancer"  # keyring服务名称
    proxy: "http://127.0.0.1:20171"  # 如果需要代理
```

### 多账户OAuth配置

```yaml
providers:
  # 账户1 - 主要账户
  - name: "Claude Code Official"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    auth_type: "auth_token"
    auth_value: "oauth"
    account_email: "your-main-account@gmail.com"  # 指定OAuth账户邮箱
    proxy: "http://127.0.0.1:20171"  # 如果需要代理
    enabled: true

  # 账户2 - 备用账户（相同name，不同account_email）
  - name: "Claude Code Official"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    auth_type: "auth_token"
    auth_value: "oauth"
    account_email: "your-backup-account@gmail.com"  # 第二个OAuth账户
    proxy: "http://127.0.0.1:20171"
    enabled: true

  # 账户3 - 第三方提供商混合使用
  - name: "Third Party Provider"
    type: "anthropic"
    base_url: "https://api.thirdparty.com"
    auth_type: "api_key"
    auth_value: "sk-your-api-key"
    enabled: true
```

### 智能模型路由配置

```yaml
model_routes:
  # 大模型优先使用主账户
  "*sonnet*":
    - provider: "Claude Code Official"
      model: "passthrough"
      priority: 1
      account_email: "your-main-account@gmail.com"  # 指定特定账户
    
    - provider: "Claude Code Official"
      model: "passthrough"
      priority: 2
      account_email: "your-backup-account@gmail.com"  # 备用账户
    
    - provider: "Third Party Provider"
      model: "passthrough"
      priority: 3
      # 无account_email，使用API Key认证

  # 小模型可以优先使用备用账户
  "*haiku*":
    - provider: "Claude Code Official"
      model: "passthrough"
      priority: 1
      account_email: "your-backup-account@gmail.com"
    
    - provider: "Claude Code Official"
      model: "passthrough"
      priority: 2
      account_email: "your-main-account@gmail.com"
```

## 授权流程

### 1. 启动服务
```bash
python src/main.py
```

### 2. 首次OAuth授权

当provider返回401错误时，系统会显示授权指令：

```
🔐 AUTHENTICATION REQUIRED - OAUTH LOGIN NEEDED
👤 Required account: your-main-account@gmail.com
================================================================================

To continue using Claude Code Provider Balancer, you need to:

1. 🌐 Open this URL in your browser:
   http://localhost:9090/oauth/generate-url

2. 🔑 Sign in with your Claude Code account
   ⚠️  Make sure to use account: your-main-account@gmail.com

3. ✅ Grant permission to the application

4. 🔄 The token will be saved automatically

5. ⚡ Retry your request - it should work now!
```

### 3. 获取授权URL

```bash
curl http://localhost:9090/oauth/generate-url
```

返回示例：
```json
{
  "authorization_url": "https://claude.ai/oauth/authorize?code=true&client_id=...",
  "expires_in": 600,
  "instructions": "Visit the URL to authorize, then exchange the code"
}
```

### 4. 完成授权

1. 访问授权URL，登录对应的Claude账户
2. 授权后复制回调URL中的 `code` 参数
3. 调用代码交换接口：

```bash
curl -X POST http://localhost:9090/oauth/exchange-code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "YOUR_AUTH_CODE",
    "account_email": "your-main-account@gmail.com"
  }'
```

### 5. 获取第二个账户的token

重复上述步骤，但使用不同的账户登录：

```bash
curl -X POST http://localhost:9090/oauth/exchange-code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "YOUR_AUTH_CODE_2", 
    "account_email": "your-backup-account@gmail.com"
  }'
```

## 管理和监控

### 查看所有OAuth账户状态

```bash
curl -s http://localhost:9090/oauth/status | jq '.'
```

返回示例：
```json
{
  "system": {
    "oauth_manager_status": "active",
    "current_time_iso": "2024-01-01 12:00:00"
  },
  "summary": {
    "total_tokens": 2,
    "healthy_tokens": 2,
    "expired_tokens": 0
  },
  "tokens": [
    {
      "account_email": "your-main-account@gmail.com",
      "is_healthy": true,
      "expires_in_human": "2小时30分钟",
      "usage_count": 127,
      "last_used": "5分钟前",
      "scopes": ["org:create_api_key", "user:profile", "user:inference"]
    },
    {
      "account_email": "your-backup-account@gmail.com",
      "is_healthy": true, 
      "expires_in_human": "1小时45分钟",
      "usage_count": 89,
      "last_used": "10分钟前",
      "scopes": ["org:create_api_key", "user:profile", "user:inference"]
    }
  ]
}
```

### 管理API接口

#### 手动刷新特定账户的token
```bash
curl -X POST http://localhost:9090/oauth/refresh/your-main-account@gmail.com
```

**注意**: Token刷新需要美国IP地址，如遇到Cloudflare拦截请使用美国代理。

#### 删除特定账户的token
```bash
curl -X DELETE http://localhost:9090/oauth/tokens/your-backup-account@gmail.com
```

#### 清除所有token
```bash
curl -X DELETE http://localhost:9090/oauth/tokens
```

#### 查看Provider状态
```bash
curl -s http://localhost:9090/providers | jq '.'
```

## 工作原理

### Provider查找逻辑

1. **精确匹配**：如果模型路由指定了`account_email`，系统会查找匹配`name`和`account_email`的provider
2. **模糊匹配**：如果没有指定`account_email`，优先匹配没有`account_email`的provider  
3. **后备机制**：如果都没有找到，返回第一个匹配`name`的provider

### Token选择逻辑

1. **指定账户**：如果provider配置了`account_email`，使用对应账户的token
2. **轮询机制**：如果没有指定账户，使用轮询策略在所有可用token中选择
3. **健康检查**：只使用未过期且健康的token

### 自动故障转移

- **for streaming requests**: 如果响应头已发送，无法故障转移（返回错误）
- **for non-streaming requests**: 总是尝试故障转移到下一个可用provider
- **当没有可用provider时**: 返回"All providers failed"错误

### 错误处理逻辑

- 如果指定的账户token不可用，请求会失败并提示需要授权
- 如果使用轮询模式，会自动跳过不健康的token
- 系统会显示需要哪个特定账户进行OAuth授权

## 高级配置

### 混合认证类型

可以将OAuth账户与API Key provider混合使用：

```yaml
providers:
  # OAuth账户1
  - name: "Claude Code Official"
    auth_type: "auth_token"
    auth_value: "oauth"
    account_email: "oauth-user1@gmail.com"
    
  # OAuth账户2
  - name: "Claude Code Official"
    auth_type: "auth_token"
    auth_value: "oauth"
    account_email: "oauth-user2@gmail.com"
    
  # API Key backup（相同name，无account_email）
  - name: "Claude Code Official" 
    auth_type: "api_key"
    auth_value: "sk-your-api-key"
    # 无account_email字段
```

### 账户特定的代理设置

```yaml
providers:
  # 美国账户，需要代理
  - name: "Claude Code Official"
    auth_value: "oauth"
    account_email: "us-account@gmail.com"
    proxy: "http://127.0.0.1:20171"
    
  # 其他地区账户，不需要代理
  - name: "Claude Code Official"
    auth_value: "oauth" 
    account_email: "other-account@gmail.com"
    # 无proxy设置
```

### 自动刷新配置

```yaml
settings:
  oauth:
    enable_auto_refresh: true  # 启用自动刷新
    proxy: "http://127.0.0.1:20171"  # 刷新时使用的代理
```

## 核心技术实现

### 自动刷新机制
- 过期前5分钟自动刷新
- 失败重试（1小时后）
- 多token独立管理
- 美国IP访问要求

### 持久化存储
- 使用系统keyring安全存储
- 重启后自动加载token
- 支持多用户环境
- 敏感数据加密保护

### 使用统计
- 自动记录每个token使用次数
- 追踪最后使用时间（人性化显示）
- 统计数据持久化存储
- 支持使用模式分析

### 安全机制
- OAuth 2.0 PKCE流程
- 状态参数防CSRF
- 最小权限原则
- Token加密存储

## 最佳实践

1. **账户分离**：使用不同账户处理不同类型的请求（如大模型vs小模型）
2. **负载均衡**：配置合理的优先级实现账户间的负载均衡
3. **监控使用**：定期检查各账户的使用情况和配额
4. **备份机制**：总是配置至少一个备用账户或API Key
5. **安全性**：确保每个账户都有独立的OAuth token，避免共享
6. **代理设置**：根据地理位置合理配置代理，确保token刷新成功

## 故障排除

### 常见问题

#### 问题：Token获取失败
- 检查account_email是否正确
- 确认OAuth授权时使用的是正确的账户  
- 验证代理设置（如果需要）
- 检查授权码是否过期（10分钟有效期）

#### 问题：找不到指定账户的provider
- 检查provider配置中的account_email字段
- 确认provider已启用（enabled: true）
- 验证模型路由中的account_email匹配

#### 问题：请求总是使用同一个账户
- 检查模型路由的优先级设置
- 确认其他账户的token是否健康
- 查看OAuth状态确认token有效性

#### 问题：Token刷新失败
- 确认网络可访问anthropic.com
- 确保使用美国IP地址或美国代理
- 检查代理配置是否正确

#### 问题：403 Cloudflare错误
- Token刷新被Cloudflare拦截
- 需要美国IP地址或美国代理
- 检查proxy配置在oauth设置中

#### 问题：重启后丢失token
- 确认`enable_persistence: true`
- 检查keyring库是否安装：`pip install keyring`
- 验证系统keyring服务可用性

### 调试日志

启用详细日志查看OAuth流程：

```yaml
settings:
  log_level: "DEBUG"  # 查看详细OAuth流程
```

### 监控命令

```bash
# 监控provider状态
watch -n 5 'curl -s http://localhost:9090/providers | jq .'

# 查看OAuth状态
watch -n 10 'curl -s http://localhost:9090/oauth/status | jq .'

# 查看实时日志
tail -f logs/logs.jsonl | jq '.'

# 过滤OAuth相关日志
tail -f logs/logs.jsonl | jq 'select(.message | contains("oauth"))'
```

## 技术架构

### 核心组件
- **oauth_manager.py** - OAuth认证管理器
- **provider_manager.py** - Provider管理增强
- **provider_auth.py** - 认证处理逻辑
- **routers/oauth.py** - OAuth API端点

### 关键数据结构
- **Provider** - 支持account_email字段的provider模型
- **ModelRoute** - 支持account_email路由的模型路由
- **TokenCredentials** - OAuth token凭据存储

### 安全考虑
- PKCE（Proof Key for Code Exchange）流程
- 状态参数防CSRF攻击
- Token安全存储和传输
- 最小权限OAuth范围

通过以上配置，你可以充分利用多个Claude账户，实现更高的可用性、更灵活的资源管理和更智能的负载均衡。