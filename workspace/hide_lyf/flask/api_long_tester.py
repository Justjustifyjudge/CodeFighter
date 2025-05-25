import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_generation():
    """测试长文本生成"""
    test_cases = [
        ("短问题", "如何实现快速排序？", 512),
        ("中等问题", "请用Python实现一个支持缓存的斐波那契数列计算函数，要求：1. 使用LRU缓存策略 2. 处理大数情况 3. 提供性能测试示例", 1024),
        ("长问题", "设计一个完整的电商系统架构，包含以下模块：\n1. 用户认证服务\n2. 商品目录服务\n3. 订单处理服务\n4. 支付网关集成\n5. 推荐系统\n6. 日志和监控系统\n\n要求：\n- 给出每个模块的详细设计\n- 说明服务间通信方式\n- 提供数据库Schema设计\n- 讨论扩展性和容错方案", 2048)
    ]
    
    for name, problem, length in test_cases:
        print(f"\n{'='*40}")
        print(f"测试案例: {name} (长度: {len(problem)})")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate_code",
            json={
                "problem": problem,
                "max_length": length,
                "difficulty": "Hard"  # 强制测试思考模型
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"生成成功! 耗时: {data['stats']['time_elapsed']:.2f}s")
            print(f"总token数: {data['stats']['final_length']}")
            print(f"内存使用: {data['stats']['memory_usage']}")
            
            # 保存结果到文件
            with open(f"result_{name}.txt", "w", encoding="utf-8") as f:
                f.write(f"问题:\n{problem}\n\n")
                if 'reasoning' in data:
                    f.write(f"分析:\n{data['reasoning']}\n\n")
                f.write(f"生成结果:\n{data['result']}")
            
            print(f"结果已保存到 result_{name}.txt")
        else:
            print(f"生成失败! 状态码: {response.status_code}")
            print(response.text)

if __name__ == '__main__':
    # 先测试健康检查
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"服务状态: {health['status']}")
    
    # 运行测试
    test_generation()