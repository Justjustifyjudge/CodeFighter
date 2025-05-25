import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# 配置
BASE_URL = "http://localhost:5000"

@dataclass
class TestResult:
    name: str
    passed: bool
    response_time: float
    error: Optional[str] = None

class AvailabilityTester:
    """仅测试系统可用性的测试套件"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.base_url = base_url
    
    def _make_request(self, endpoint: str, payload: Dict = None, method: str = "POST") -> Tuple[bool, Dict]:
        """通用请求方法"""
        try:
            start_time = time.time()
            if method == "POST":
                resp = self.session.post(endpoint, json=payload)
            else:
                resp = self.session.get(endpoint)
            elapsed = time.time() - start_time
            
            # 只要返回了200状态码就认为成功
            return True, {
                "status_code": resp.status_code,
                "response_time": elapsed
            }
        except Exception as e:
            return False, {
                "error": str(e),
                "response_time": 0
            }
    
    def test_health_check(self) -> TestResult:
        """测试健康检查端点"""
        endpoint = f"{self.base_url}/health"
        success, result = self._make_request(endpoint, method="GET")
        
        return TestResult(
            name="健康检查",
            passed=success and result["status_code"] == 200,
            response_time=result.get("response_time", 0),
            error=None if success else result.get("error")
        )
    
    def test_basic_workflow(self) -> List[TestResult]:
        """测试基本工作流程"""
        endpoints = [
            ("/predict_difficulty", {"problem": "如何计算两个数的和？"}),
            ("/generate_code", {"problem": "如何用Python实现快速排序？"}),
            ("/generate_code", {"problem": "简单问题", "difficulty": "Easy"}),
            ("/generate_code", {"problem": "困难问题", "difficulty": "Hard"})
        ]
        
        results = []
        for path, payload in endpoints:
            endpoint = f"{self.base_url}{path}"
            success, result = self._make_request(endpoint, payload=payload)
            
            results.append(TestResult(
                name=f"基本流程测试[{path}]",
                passed=success and result["status_code"] == 200,
                response_time=result.get("response_time", 0),
                error=None if success else f"状态码{result.get('status_code')}"
            ))
        
        return results
    
    def test_edge_cases(self) -> List[TestResult]:
        """测试边界情况"""
        test_cases = [
            ("长文本输入", {"problem": "长文本" * 1000}),
            ("特殊字符", {"problem": "!@#$%^&*()"}),
            ("最小参数", {"problem": "test"}),
            ("完整参数", {
                "problem": "test", 
                "difficulty": "Medium",
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9
            })
        ]
        
        results = []
        for name, payload in test_cases:
            endpoint = f"{self.base_url}/generate_code"
            success, result = self._make_request(endpoint, payload=payload)
            
            results.append(TestResult(
                name=f"边界情况[{name}]",
                passed=success and result["status_code"] == 200,
                response_time=result.get("response_time", 0),
                error=None if success else f"状态码{result.get('status_code')}"
            ))
        
        return results
    
    def test_error_handling(self) -> List[TestResult]:
        """测试错误处理"""
        test_cases = [
            ("缺失problem参数", {"max_length": 100}, 400),
            ("空问题文本", {"problem": ""}, 500),
            ("无效难度级别", {"problem": "test", "difficulty": "Invalid"}, 400)
        ]
        
        results = []
        for name, payload, expected_code in test_cases:
            endpoint = f"{self.base_url}/generate_code"
            success, result = self._make_request(endpoint, payload=payload)
            
            # 验证返回了预期的错误状态码
            results.append(TestResult(
                name=f"错误处理[{name}]",
                passed=success and result["status_code"] == expected_code,
                response_time=result.get("response_time", 0),
                error=None if (success and result["status_code"] == expected_code) 
                          else f"预期{expected_code}，得到{result.get('status_code')}"
            ))
        
        return results
    
    def run_availability_test(self) -> Dict[str, Any]:
        """运行可用性测试套件"""
        print("=== 开始系统可用性测试 ===")
        
        # 运行所有测试
        health_test = self.test_health_check()
        workflow_tests = self.test_basic_workflow()
        edge_case_tests = self.test_edge_cases()
        error_tests = self.test_error_handling()
        
        # 汇总结果
        all_tests = [health_test] + workflow_tests + edge_case_tests + error_tests
        passed = sum(1 for t in all_tests if t.passed)
        total = len(all_tests)
        
        # 打印结果摘要
        print("\n=== 测试结果摘要 ===")
        for test in all_tests:
            status = "✓" if test.passed else "×"
            print(f"{status} {test.name} ({test.response_time:.2f}s)")
        
        # 返回汇总结果
        return {
            "total_tests": total,
            "passed_tests": passed,
            "success_rate": passed / total,
            "health_check": health_test.passed,
            "basic_workflow": all(t.passed for t in workflow_tests),
            "edge_cases": all(t.passed for t in edge_case_tests),
            "error_handling": all(t.passed for t in error_tests)
        }

if __name__ == "__main__":
    tester = AvailabilityTester()
    results = tester.run_availability_test()
    
    print("\n=== 测试总结 ===")
    print(f"总测试数: {results['total_tests']}")
    print(f"通过数: {results['passed_tests']}")
    print(f"成功率: {results['success_rate']*100:.1f}%")
    
    if results["success_rate"] == 1:
        print("\n✓ 系统可用性测试全部通过！")
    else:
        print("\n× 部分测试失败，请检查错误信息")