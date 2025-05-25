import requests
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# 配置
BASE_URL = "http://localhost:5000"
GENERATE_ENDPOINT = f"{BASE_URL}/generate_code"
DIFFICULTY_ENDPOINT = f"{BASE_URL}/predict_difficulty"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

# 测试数据
TEST_PROBLEMS = {
    "easy": "给定一个整数数组，找出所有元素的和",
    "medium": "实现一个函数，检查二叉树是否是平衡的",
    "hard": "设计一个支持以下操作的数据结构：insert, delete, getRandom，所有操作应在平均O(1)时间内完成"
}

@dataclass
class TestResult:
    name: str
    passed: bool
    response_time: float
    details: Dict[str, Any]
    error: Optional[str] = None

class EnhancedAPITester:
    """增强版API测试套件，包含显式难度和思考模型测试"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.base_url = base_url
        self.endpoints = {
            "health": f"{base_url}/health",
            "generate": f"{base_url}/generate_code",
            "difficulty": f"{base_url}/predict_difficulty"
        }
    
    def _make_request(self, endpoint: str, payload: Dict = None, method: str = "POST") -> Tuple[bool, Dict]:
        """通用请求方法"""
        try:
            start_time = time.time()
            if method == "POST":
                resp = self.session.post(endpoint, json=payload)
            else:
                resp = self.session.get(endpoint)
            elapsed = time.time() - start_time
            
            return True, {
                "status_code": resp.status_code,
                "data": resp.json() if resp.content else {},
                "response_time": elapsed
            }
        except Exception as e:
            return False, {
                "error": str(e),
                "response_time": 0
            }
    
    def test_health_check(self) -> TestResult:
        """测试健康检查端点"""
        success, result = self._make_request(self.endpoints["health"], method="GET")
        
        return TestResult(
            name="健康检查",
            passed=success and result["status_code"] == 200,
            response_time=result.get("response_time", 0),
            details=result,
            error=None if success else result.get("error")
        )
    
    def test_difficulty_prediction(self) -> List[TestResult]:
        """测试难度预测端点"""
        results = []
        for level, problem in TEST_PROBLEMS.items():
            success, result = self._make_request(
                self.endpoints["difficulty"],
                payload={"problem": problem}
            )
            
            test_passed = (success and 
                         result["status_code"] == 200 and 
                         "difficulty" in result.get("data", {}))
            
            results.append(TestResult(
                name=f"难度预测[{level}]",
                passed=test_passed,
                response_time=result.get("response_time", 0),
                details=result,
                error=None if test_passed else "格式验证失败"
            ))
        
        return results
    
    def test_code_generation(self) -> List[TestResult]:
        """测试代码生成端点（基础功能）"""
        results = []
        for level, problem in TEST_PROBLEMS.items():
            success, result = self._make_request(
                self.endpoints["generate"],
                payload={
                    "problem": problem,
                    "max_length": 1024
                }
            )
            
            # 验证基本返回格式和思考模型结果（如果是Hard问题）
            data = result.get("data", {})
            is_hard = data.get("difficulty", "").lower() == "hard"
            
            test_passed = (success and 
                         result["status_code"] == 200 and 
                         all(k in data for k in ["result", "difficulty", "input_length"]) and
                         (not is_hard or "reasoning" in data))
            
            results.append(TestResult(
                name=f"代码生成[{level}]",
                passed=test_passed,
                response_time=result.get("response_time", 0),
                details=result,
                error=None if test_passed else "格式验证失败"
            ))
        
        return results
    
    def test_explicit_difficulty(self) -> List[TestResult]:
        """测试显式指定难度级别"""
        test_cases = [
            ("显式Easy", "Easy", False),
            ("显式Medium", "Medium", False),
            ("显式Hard", "Hard", True),
            ("显式小写hard", "hard", True),
            ("显式大写HARD", "HARD", True)
        ]
        
        results = []
        for name, difficulty, expect_reasoning in test_cases:
            success, result = self._make_request(
                self.endpoints["generate"],
                payload={
                    "problem": TEST_PROBLEMS["easy"],  # 使用简单问题测试显式难度覆盖
                    "difficulty": difficulty
                }
            )
            
            data = result.get("data", {})
            test_passed = (success and 
                          result["status_code"] == 200 and
                          data.get("difficulty", "").capitalize() == difficulty.capitalize() and
                          data.get("is_explicit_difficulty", False) and
                          ("reasoning" in data) == expect_reasoning)
            
            results.append(TestResult(
                name=f"显式难度[{name}]",
                passed=test_passed,
                response_time=result.get("response_time", 0),
                details=result,
                error=None if test_passed else "显式难度验证失败"
            ))
        
        return results
    
    def test_invalid_difficulty(self) -> TestResult:
        """测试无效难度级别"""
        success, result = self._make_request(
            self.endpoints["generate"],
            payload={
                "problem": TEST_PROBLEMS["easy"],
                "difficulty": "InvalidLevel"
            }
        )
        
        test_passed = (success and 
                      result["status_code"] == 400 and
                      "Invalid difficulty level" in str(result.get("data", {})))
        
        return TestResult(
            name="无效难度测试",
            passed=test_passed,
            response_time=result.get("response_time", 0),
            details=result,
            error=None if test_passed else "预期400错误未触发"
        )
    
    def test_robustness(self) -> List[TestResult]:
        """测试API鲁棒性"""
        test_cases = [
            # 预期返回200的情况
            ("超大输入", {"problem": "a"*100000, "temperature": 0.5}, 200),
            ("无效温度", {"problem": "test", "temperature": 2.0}, 200),
            ("极高温度", {"problem": "test", "temperature": 100.0}, 200),
            ("超大max_length", {"problem": "test", "max_length": 100000}, 200),
            ("HTML内容", {"problem": "<html><body>test</body></html>"}, 200),
            ("特殊字符", {"problem": "!@#$%^&*()"}, 200),
            ("仅空格", {"problem": "    "}, 200),
            ("混合参数", {"problem": "test", "temperature": 0.5, "top_p": 1.5, "unknown_param": "value"}, 200),
            ("显式难度+无效参数", {"problem": "test", "difficulty": "Hard", "invalid_param": "value"}, 200),
            
            # 预期返回500的情况
            ("负温度", {"problem": "test", "temperature": -1.0}, 500),
            ("空问题", {"problem": ""}, 500),
            
            # 预期返回400的情况
            ("缺失问题", {"max_length": 100}, 400),
            ("无效难度", {"problem": "test", "difficulty": "WrongLevel"}, 400)
        ]
        
        results = []
        for name, payload, expected_code in test_cases:
            success, result = self._make_request(
                self.endpoints["generate"],
                payload=payload
            )
            
            test_passed = success and result["status_code"] == expected_code
            
            results.append(TestResult(
                name=f"鲁棒性[{name}]",
                passed=test_passed,
                response_time=result.get("response_time", 0),
                details=result,
                error=None if test_passed else f"预期{expected_code}，得到{result.get('status_code')}"
            ))
        
        return results
    
    def test_reasoning_quality(self) -> TestResult:
        """测试思考模型生成的分析质量（基础验证）"""
        success, result = self._make_request(
            self.endpoints["generate"],
            payload={
                "problem": TEST_PROBLEMS["hard"],
                "difficulty": "Hard"  # 强制使用思考模型
            }
        )
        
        data = result.get("data", {})
        reasoning = data.get("reasoning", "")
        
        test_passed = (success and 
                      result["status_code"] == 200 and
                      len(reasoning) > 50  # 简单验证分析内容长度
                      # "分析" in reasoning and  # 验证包含关键词
                      # data.get("difficulty") == "Hard")
                      )
        
        return TestResult(
            name="思考模型质量",
            passed=test_passed,
            response_time=result.get("response_time", 0),
            details={"reasoning_length": len(reasoning)},
            error=None if test_passed else "分析内容不符合预期"
        )
    
    def run_performance_test(self, n: int = 5) -> TestResult:
        """运行性能基准测试"""
        times = []
        reasoning_times = []
        successes = 0
        
        for i in range(n):
            # 交替测试简单和困难问题
            level = "hard" if i % 2 == 0 else "easy"
            payload = {
                "problem": TEST_PROBLEMS[level],
                "max_length": 512
            }
            
            if level == "hard":
                payload["difficulty"] = "Hard"  # 强制使用思考模型
            
            success, result = self._make_request(
                self.endpoints["generate"],
                payload=payload
            )
            
            if success and result["status_code"] == 200:
                times.append(result["response_time"])
                successes += 1
                
                # 记录思考模型处理时间（如果是Hard问题）
                data = result.get("data", {})
                if data.get("difficulty") == "Hard":
                    reasoning_times.append(result["response_time"])
        
        stats = {
            "total_requests": n,
            "successful_requests": successes,
            "success_rate": successes / n,
            "average_time": sum(times) / len(times) if times else 0,
            "max_time": max(times) if times else 0,
            "min_time": min(times) if times else 0,
            "average_reasoning_time": sum(reasoning_times) / len(reasoning_times) if reasoning_times else 0
        }
        
        return TestResult(
            name="性能测试",
            passed=successes == n,
            response_time=stats["average_time"],
            details=stats,
            error=None if successes == n else f"成功率{stats['success_rate']*100}%"
        )
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """运行完整的测试套件"""
        print("=== 开始增强版API测试套件 ===")
        
        # 运行所有测试
        health_test = self.test_health_check()
        difficulty_tests = self.test_difficulty_prediction()
        generation_tests = self.test_code_generation()
        explicit_difficulty_tests = self.test_explicit_difficulty()
        invalid_diff_test = self.test_invalid_difficulty()
        robustness_tests = self.test_robustness()
        reasoning_test = self.test_reasoning_quality()
        perf_test = self.run_performance_test()
        
        # 汇总结果
        all_tests = ([health_test] + difficulty_tests + generation_tests + 
                    explicit_difficulty_tests + [invalid_diff_test] + 
                    robustness_tests + [reasoning_test] + [perf_test])
        passed = sum(1 for t in all_tests if t.passed)
        total = len(all_tests)
        
        # 打印详细结果
        print("\n=== 详细测试结果 ===")
        for test in all_tests:
            status = "✓ 通过" if test.passed else "× 失败"
            print(f"{test.name}: {status} ({test.response_time:.2f}s)")
            if test.error:
                print(f"    -> 错误: {test.error}")
        
        # 打印性能数据
        print("\n=== 性能数据 ===")
        print(f"平均响应时间: {perf_test.details['average_time']:.2f}s")
        print(f"思考模型平均时间: {perf_test.details['average_reasoning_time']:.2f}s")
        print(f"最慢响应: {perf_test.details['max_time']:.2f}s")
        print(f"最快响应: {perf_test.details['min_time']:.2f}s")
        print(f"成功率: {perf_test.details['success_rate']*100:.1f}%")
        
        # 返回汇总结果
        return {
            "total_tests": total,
            "passed_tests": passed,
            "success_rate": passed / total,
            "health_check": health_test.passed,
            "difficulty_prediction": all(t.passed for t in difficulty_tests),
            "code_generation": all(t.passed for t in generation_tests),
            "explicit_difficulty": all(t.passed for t in explicit_difficulty_tests),
            "invalid_difficulty": invalid_diff_test.passed,
            "robustness": all(t.passed for t in robustness_tests),
            "reasoning_quality": reasoning_test.passed,
            "performance": perf_test.passed
        }

if __name__ == "__main__":
    tester = EnhancedAPITester()
    results = tester.run_full_test_suite()
    
    print("\n=== 测试总结 ===")
    print(f"总测试数: {results['total_tests']}")
    print(f"通过数: {results['passed_tests']}")
    print(f"成功率: {results['success_rate']*100:.1f}%")
    
    # 打印各模块测试结果
    print("\n=== 模块测试结果 ===")
    print(f"健康检查: {'✓' if results['health_check'] else '×'}")
    print(f"难度预测: {'✓' if results['difficulty_prediction'] else '×'}")
    print(f"代码生成: {'✓' if results['code_generation'] else '×'}")
    print(f"显式难度: {'✓' if results['explicit_difficulty'] else '×'}")
    print(f"无效难度: {'✓' if results['invalid_difficulty'] else '×'}")
    print(f"鲁棒性: {'✓' if results['robustness'] else '×'}")
    print(f"思考模型: {'✓' if results['reasoning_quality'] else '×'}")
    print(f"性能测试: {'✓' if results['performance'] else '×'}")
    
    if results["success_rate"] == 1:
        print("\n✓ 所有测试通过！")
    else:
        print("\n× 部分测试失败，请检查错误信息")