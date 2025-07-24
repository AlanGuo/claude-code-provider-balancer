#!/usr/bin/env python3
"""
测试客户端连接断开处理
包括流式请求过程中客户端断开连接的场景
"""

import json
import time
import requests
import sys
import os
import threading
import signal
import socket
from contextlib import contextmanager

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import get_claude_code_headers

BASE_URL = "http://localhost:9090"

class TestClientDisconnect:
    def __init__(self):
        self.base_url = BASE_URL
        self.headers = get_claude_code_headers()
        
    def test_stream_early_disconnect(self):
        """测试流式请求早期断开"""
        print("测试: 流式请求早期断开")
        
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "请从1数到50，每个数字单独输出"}],
            "max_tokens": 200,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            assert response.status_code == 200, f"请求失败: {response.status_code}"
            
            chunks_received = 0
            
            # 只接收几个数据块就断开
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_str)
                            if chunk_data.get("type") == "content_block_delta":
                                chunks_received += 1
                                print(f"   收到数据块 {chunks_received}")
                                
                                # 收到几个数据块后主动断开
                                if chunks_received >= 3:
                                    print("   主动断开连接...")
                                    response.close()
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
            
            if chunks_received > 0:
                print(f"✅ 早期断开测试通过 (收到 {chunks_received} 个数据块后断开)")
                return True
            else:
                print("⚠️  未收到任何数据")
                return False
                
        except Exception as e:
            print(f"❌ 早期断开测试失败: {e}")
            return False
    
    def test_stream_mid_response_disconnect(self):
        """测试流式响应中途断开"""
        print("测试: 流式响应中途断开")
        
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "数数字从1到20，每个数字单独说"}],
            "max_tokens": 200,
            "stream": True
        }
        
        disconnect_occurred = False
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            assert response.status_code == 200, f"请求失败: {response.status_code}"
            
            chunks_received = 0
            content_chunks = []
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_str)
                            if chunk_data.get("type") == "content_block_delta":
                                delta = chunk_data.get("delta", {})
                                if "text" in delta:
                                    content_chunks.append(delta["text"])
                                    chunks_received += 1
                                    
                                    # 在响应中途断开（当收到足够内容时）
                                    current_content = "".join(content_chunks)
                                    if len(current_content) > 50 or chunks_received > 10:
                                        print(f"   在第 {chunks_received} 个数据块后断开")
                                        print(f"   已收到内容: {current_content[:50]}...")
                                        response.close()
                                        disconnect_occurred = True
                                        break
                                        
                        except json.JSONDecodeError:
                            continue
            
            if disconnect_occurred and chunks_received > 0:
                print(f"✅ 中途断开测试通过 (收到 {chunks_received} 个数据块)")
                return True
            elif chunks_received > 0:
                print("ℹ️  响应完成，未触发中途断开")
                return True
            else:
                print("⚠️  未收到任何数据")
                return False
                
        except Exception as e:
            print(f"❌ 中途断开测试失败: {e}")
            return False
    
    def test_concurrent_disconnect(self):
        """测试并发请求中的断开"""
        print("测试: 并发请求中的断开")
        
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "并发断开测试"}],
            "max_tokens": 100,
            "stream": True
        }
        
        results = []
        
        def stream_request_with_disconnect(request_id, disconnect_after_chunks=5):
            """发送流式请求并在指定数据块后断开"""
            try:
                # 为每个请求使用不同的内容避免去重
                unique_payload = {
                    "model": "claude-3-5-haiku-20241022",
                    "messages": [{"role": "user", "content": f"并发断开测试 - 请求{request_id}"}],
                    "max_tokens": 100,
                    "stream": True
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=unique_payload,
                    stream=True,
                    timeout=30
                )
                
                if response.status_code != 200:
                    results.append({
                        "id": request_id,
                        "success": False,
                        "error": f"Status code: {response.status_code}"
                    })
                    return
                
                chunks_received = 0
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            
                            if data_str.strip() == '[DONE]':
                                break
                                
                            try:
                                chunk_data = json.loads(data_str)
                                if chunk_data.get("type") == "content_block_delta":
                                    chunks_received += 1
                                    
                                    if chunks_received >= disconnect_after_chunks:
                                        response.close()
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
                
                results.append({
                    "id": request_id,
                    "success": True,
                    "chunks_received": chunks_received,
                    "disconnected_at": disconnect_after_chunks
                })
                
            except Exception as e:
                results.append({
                    "id": request_id,
                    "success": False,
                    "error": str(e)
                })
        
        try:
            # 创建多个并发流式请求，在不同时间点断开
            threads = []
            disconnect_points = [2, 3, 5, 7, 10]  # 不同的断开点
            
            for i, disconnect_point in enumerate(disconnect_points):
                thread = threading.Thread(
                    target=stream_request_with_disconnect,
                    args=(i, disconnect_point)
                )
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=60)
            
            # 分析结果
            successful = sum(1 for r in results if r["success"])
            total = len(results)
            
            print(f"   并发断开测试结果: {successful}/{total} 成功")
            
            for result in results:
                if result["success"]:
                    chunks = result["chunks_received"]
                    disconnect_at = result["disconnected_at"]
                    print(f"   请求 {result['id']}: 收到 {chunks} 个数据块，计划在第 {disconnect_at} 个断开")
                else:
                    print(f"   请求 {result['id']}: 失败 - {result['error']}")
            
            if successful > 0:
                print("✅ 并发断开测试通过")
                return True
            else:
                print("❌ 所有并发请求都失败")
                return False
                
        except Exception as e:
            print(f"❌ 并发断开测试失败: {e}")
            return False
    
    def test_abrupt_connection_close(self):
        """测试突然关闭连接"""
        print("测试: 突然关闭连接")
        
        payload = {
            "model": "claude-3-5-haiku-20241022", 
            "messages": [{"role": "user", "content": "测试突然断开"}],
            "max_tokens": 50,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            assert response.status_code == 200, f"请求失败: {response.status_code}"
            
            # 立即关闭连接，不读取任何数据
            response.close()
            
            print("✅ 突然关闭连接测试通过")
            
            # 验证服务器仍能处理新请求
            time.sleep(1)
            
            test_payload = {
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "验证服务器状态"}],
                "max_tokens": 10,
                "stream": False
            }
            
            verify_response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=test_payload,
                timeout=30
            )
            
            if verify_response.status_code == 200:
                print("✅ 服务器在连接断开后仍正常工作")
                return True
            else:
                print(f"⚠️  服务器状态异常: {verify_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 突然关闭连接测试失败: {e}")
            return False
    
    def test_timeout_vs_disconnect(self):
        """测试超时与主动断开的区别"""
        print("测试: 超时与主动断开的区别")
        
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "请写一个详细的技术文档，包含多个章节"}],
            "max_tokens": 500,
            "stream": True
        }
        
        try:
            # 测试1: 主动断开
            print("   测试主动断开...")
            response1 = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            chunks_before_disconnect = 0
            for line in response1.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        chunks_before_disconnect += 1
                        if chunks_before_disconnect >= 3:
                            response1.close()
                            break
            
            print(f"   主动断开: 收到 {chunks_before_disconnect} 个数据块")
            
            # 测试2: 超时
            print("   测试超时...")
            try:
                response2 = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=2  # 很短的超时
                )
                
                chunks_before_timeout = 0
                for line in response2.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            chunks_before_timeout += 1
                
                print(f"   意外完成: 收到 {chunks_before_timeout} 个数据块")
                
            except requests.exceptions.Timeout:
                print("   正确触发超时")
            
            print("✅ 超时与断开区别测试完成")
            return True
            
        except Exception as e:
            print(f"❌ 超时与断开测试失败: {e}")
            return False
    
    def test_rapid_connect_disconnect(self):
        """测试快速连接和断开"""
        print("测试: 快速连接和断开")
        
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "快速测试"}],
            "max_tokens": 50,
            "stream": True
        }
        
        successful_connects = 0
        
        try:
            # 快速创建和关闭多个连接
            for i in range(5):
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/messages",
                        headers=self.headers,
                        json=payload,
                        stream=True,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        successful_connects += 1
                        
                        # 读取少量数据后立即关闭
                        chunks = 0
                        for line in response.iter_lines():
                            if line:
                                chunks += 1
                                if chunks >= 2:
                                    break
                        
                        response.close()
                        print(f"   连接 {i+1}: 成功 (收到 {chunks} 个数据块)")
                    else:
                        print(f"   连接 {i+1}: 失败 - {response.status_code}")
                        
                except Exception as e:
                    print(f"   连接 {i+1}: 异常 - {e}")
                
                # 短暂延迟
                time.sleep(0.1)
            
            print(f"   快速连接测试: {successful_connects}/5 成功")
            
            if successful_connects > 0:
                print("✅ 快速连接断开测试通过")
                return True
            else:
                print("❌ 所有快速连接都失败")
                return False
                
        except Exception as e:
            print(f"❌ 快速连接断开测试失败: {e}")
            return False
    
    def test_duplicate_requests_after_original_disconnect(self):
        """测试原始请求断开后重复请求仍能正常工作"""
        print("测试: 原始请求断开后重复请求继续工作")
        
        # 使用相同的payload确保触发去重机制，内容要足够长以产生流式响应
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "请详细写一篇超长的文章，至少2000字，关于以下所有主题：1) Python编程语言的完整历史从1989年至今 2) Python的所有核心特性和语法 3) 面向对象编程的详细解释 4) 函数式编程概念 5) Python的标准库介绍 6) Web开发框架比较 7) 数据科学和机器学习应用 8) 并发和异步编程 9) 测试和调试最佳实践 10) Python的未来发展趋势。每个主题都要详细展开，给出具体例子和代码示例。"}],
            "max_tokens": 2000,
            "stream": True
        }
        
        original_chunks = []
        duplicate_chunks = []
        original_disconnected = False
        duplicate_completed = False
        
        def original_request():
            """原始请求 - 会在开始后很快断开"""
            nonlocal original_chunks, original_disconnected
            try:
                response = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    chunks_received = 0
                    # 简化逻辑：收到响应后立即延迟一下再断开
                    import time
                    time.sleep(0.1)  # 让服务端开始处理
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            print(f"   原始请求收到行: {line_str[:50]}")
                            chunks_received += 1
                            
                            # 收到任何数据后立即断开
                            if chunks_received >= 1:
                                print(f"   原始请求在收到 {chunks_received} 行后断开")
                                response.close()
                                original_disconnected = True
                                break
                            
            except Exception as e:
                print(f"   原始请求异常: {e}")
                original_disconnected = True
        
        def duplicate_request():
            """重复请求 - 应该能获得完整响应"""
            nonlocal duplicate_chunks, duplicate_completed
            try:
                # 稍微延迟确保原始请求先发送
                time.sleep(0.5)
                
                response = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload,  # 完全相同的payload
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    chunks_received = 0
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if chunk_data.get("type") == "content_block_delta":
                                        duplicate_chunks.append(chunk_data)
                                        chunks_received += 1
                                    elif chunk_data.get("type") == "message_stop":
                                        duplicate_completed = True
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    print(f"   重复请求收到 {chunks_received} 个数据块")
                            
            except Exception as e:
                print(f"   重复请求异常: {e}")
        
        try:
            # 启动两个并发请求
            original_thread = threading.Thread(target=original_request)
            duplicate_thread = threading.Thread(target=duplicate_request)
            
            original_thread.start()
            duplicate_thread.start()
            
            # 等待两个请求完成
            original_thread.join(timeout=60)
            duplicate_thread.join(timeout=60)
            
            # 分析结果
            original_count = len(original_chunks)
            duplicate_count = len(duplicate_chunks)
            
            print(f"   原始请求收到: {original_count} 个数据块")
            print(f"   重复请求收到: {duplicate_count} 个数据块")
            print(f"   原始请求断开: {original_disconnected}")
            print(f"   重复请求完成: {duplicate_completed}")
            
            # 验证逻辑：
            # 1. 原始请求应该断开
            # 2. 重复请求应该收到更多或相等的数据块（因为provider继续收集数据）
            # 3. 重复请求应该能正常完成
            
            if (original_disconnected and 
                duplicate_count >= original_count and 
                duplicate_count > 0):
                print("✅ 重复请求在原始断开后正常工作测试通过")
                return True
            else:
                print("❌ 重复请求未能在原始断开后正常工作")
                print(f"   期望: 原始断开={original_disconnected}, 重复数据>={original_count}")
                print(f"   实际: 重复数据={duplicate_count}")
                return False
                
        except Exception as e:
            print(f"❌ 重复请求测试失败: {e}")
            return False
    
    def test_multiple_duplicates_after_original_disconnect(self):
        """测试原始请求断开后多个重复请求都能正常工作"""
        print("测试: 原始请求断开后多个重复请求工作")
        
        # 使用相同的payload确保触发去重机制，内容要足够长以产生流式响应
        payload = {
            "model": "claude-3-5-haiku-20241022", 
            "messages": [{"role": "user", "content": "多重复请求测试：请写一篇关于机器学习基础概念的详细文章，包括监督学习、无监督学习、强化学习的区别和应用"}],
            "max_tokens": 600,
            "stream": True
        }
        
        results = {}
        
        def make_request(request_id, disconnect_early=False):
            """发送请求"""
            nonlocal results
            try:
                # 为原始请求添加延迟确保它先启动
                if request_id == "original":
                    pass  # 立即启动
                else:
                    time.sleep(0.3)  # 重复请求稍后启动
                
                response = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    chunks_received = 0
                    disconnected = False
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if chunk_data.get("type") == "content_block_delta":
                                        chunks_received += 1
                                        
                                        # 原始请求在中途断开
                                        if disconnect_early and chunks_received >= 1:
                                            print(f"   {request_id} 在第 {chunks_received} 个数据块后断开")
                                            response.close()
                                            disconnected = True
                                            # 增加延迟让服务端检测到断开
                                            import time
                                            time.sleep(0.1)
                                            break
                                            
                                except json.JSONDecodeError:
                                    continue
                    
                    results[request_id] = {
                        "chunks": chunks_received,
                        "disconnected": disconnected,
                        "success": True
                    }
                else:
                    results[request_id] = {
                        "chunks": 0,
                        "disconnected": False,
                        "success": False,
                        "error": f"Status: {response.status_code}"
                    }
                            
            except Exception as e:
                results[request_id] = {
                    "chunks": 0,
                    "disconnected": True,
                    "success": False,
                    "error": str(e)
                }
        
        try:
            # 创建多个请求：1个原始请求（会断开）+ 3个重复请求
            threads = []
            
            # 原始请求 - 会断开
            original_thread = threading.Thread(
                target=make_request, 
                args=("original", True)
            )
            threads.append(original_thread)
            
            # 3个重复请求 - 不会断开
            for i in range(3):
                duplicate_thread = threading.Thread(
                    target=make_request,
                    args=(f"duplicate_{i}", False)
                )
                threads.append(duplicate_thread)
            
            # 启动所有请求
            for thread in threads:
                thread.start()
            
            # 等待所有请求完成
            for thread in threads:
                thread.join(timeout=60)
            
            # 分析结果
            print("   请求结果:")
            for req_id, result in results.items():
                if result["success"]:
                    status = "断开" if result["disconnected"] else "完成"
                    print(f"     {req_id}: {result['chunks']} 个数据块 ({status})")
                else:
                    print(f"     {req_id}: 失败 - {result.get('error', '未知错误')}")
            
            # 验证逻辑
            original = results.get("original", {})
            duplicates = [results.get(f"duplicate_{i}", {}) for i in range(3)]
            
            # 原始请求应该断开
            original_disconnected = original.get("disconnected", False)
            original_chunks = original.get("chunks", 0)
            
            # 重复请求应该收到数据且数量 >= 原始请求
            successful_duplicates = 0
            for dup in duplicates:
                if (dup.get("success", False) and 
                    dup.get("chunks", 0) >= original_chunks and
                    dup.get("chunks", 0) > 0):
                    successful_duplicates += 1
            
            if original_disconnected and successful_duplicates >= 2:
                print(f"✅ 多重复请求测试通过 ({successful_duplicates}/3 重复请求成功)")
                return True
            else:
                print(f"❌ 多重复请求测试失败")
                print(f"   原始断开: {original_disconnected}, 成功重复: {successful_duplicates}/3")
                return False
                
        except Exception as e:
            print(f"❌ 多重复请求测试失败: {e}")
            return False

    def test_server_recovery_after_disconnects(self):
        """测试断开后服务器恢复"""
        print("测试: 断开后服务器恢复")
        
        try:
            # 先进行一些断开操作
            print("   执行多次断开操作...")
            
            payload = {
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "恢复测试"}],
                "max_tokens": 30,
                "stream": True
            }
            
            # 执行几次快速断开
            for i in range(3):
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/messages",
                        headers=self.headers,
                        json=payload,
                        stream=True,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        # 立即断开
                        response.close()
                        print(f"   断开操作 {i+1} 完成")
                    
                except:
                    pass
                
                time.sleep(0.5)
            
            # 等待一段时间
            print("   等待服务器恢复...")
            time.sleep(2)
            
            # 测试服务器是否能正常处理新请求
            normal_payload = {
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "正常请求测试"}],
                "max_tokens": 20,
                "stream": False
            }
            
            recovery_response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=normal_payload,
                timeout=30
            )
            
            if recovery_response.status_code == 200:
                data = recovery_response.json()
                if "content" in data:
                    print("✅ 服务器成功恢复，能正常处理请求")
                    return True
                else:
                    print("⚠️  服务器响应格式异常")
                    return False
            else:
                print(f"❌ 服务器恢复失败: {recovery_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 服务器恢复测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始运行 Client Disconnect 测试套件")
        print("=" * 60)
        
        tests = [
            self.test_stream_early_disconnect,
            self.test_stream_mid_response_disconnect,
            self.test_concurrent_disconnect,
            self.test_abrupt_connection_close,
            self.test_timeout_vs_disconnect,
            self.test_rapid_connect_disconnect,
            self.test_duplicate_requests_after_original_disconnect,
            self.test_multiple_duplicates_after_original_disconnect,
            self.test_server_recovery_after_disconnects
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ 测试执行异常: {e}")
            print("-" * 40)
        
        print(f"\n测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过!")
            return True
        else:
            print(f"⚠️  {total - passed} 个测试失败")
            return False

def main():
    """主函数"""
    # 检查服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ 服务器未正常运行，状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ 无法连接到服务器，请确保服务器正在运行")
        print("   启动命令: python src/main.py")
        return False
    
    # 运行测试
    tester = TestClientDisconnect()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)