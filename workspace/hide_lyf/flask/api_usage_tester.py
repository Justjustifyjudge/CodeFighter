import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pprint import pprint

# 配置
BASE_URL = "http://localhost:5000"

@dataclass
class TestResult:
    name: str
    passed: bool
    response_time: float
    response_data: Optional[Dict] = None  # 新增：存储完整响应数据
    error: Optional[str] = None

class AvailabilityTester:
    """系统可用性测试套件（带详细响应数据打印）"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.base_url = base_url
    
    def _make_request(self, endpoint: str, payload: Dict = None, method: str = "POST") -> Tuple[bool, Dict]:
        """通用请求方法（返回完整响应数据）"""
        try:
            start_time = time.time()
            if method == "POST":
                resp = self.session.post(endpoint, json=payload)
            else:
                resp = self.session.get(endpoint)
            elapsed = time.time() - start_time
            
            # 返回完整响应数据
            return True, {
                "status_code": resp.status_code,
                "response_time": elapsed,
                "data": resp.json() if resp.content else None,
                "request_payload": payload
            }
        except Exception as e:
            return False, {
                "error": str(e),
                "response_time": 0,
                "data": None,
                "request_payload": payload
            }
    
    def test_health_check(self) -> TestResult:
        """测试健康检查端点"""
        endpoint = f"{self.base_url}/health"
        success, result = self._make_request(endpoint, method="GET")
        
        return TestResult(
            name="健康检查",
            passed=success and result["status_code"] == 200,
            response_time=result.get("response_time", 0),
            response_data=result,
            error=None if success else result.get("error")
        )
    
    def test_basic_workflow(self) -> List[TestResult]:
        """测试基本工作流程"""
        test_cases = [
            ("预测难度", "/predict_difficulty", {"problem": "如何计算两个数的和？"}),
            ("自动生成代码", "/generate_code", {"problem": "如何用Python实现快速排序？"}),
            ("显式Easy难度", "/generate_code", {"problem": "杨辉三角如何打印", "difficulty": "Easy"}),
            ("显式Hard难度", "/generate_code", {"problem": "请你实现红黑树", "difficulty": "Hard"})
        ]
        
        results = []
        for name, path, payload in test_cases:
            endpoint = f"{self.base_url}{path}"
            success, result = self._make_request(endpoint, payload=payload)
            
            results.append(TestResult(
                name=f"基本流程[{name}]",
                passed=success and result["status_code"] == 200,
                response_time=result.get("response_time", 0),
                response_data=result,
                error=None if success else f"状态码{result.get('status_code')}"
            ))
        
        return results
    
    def run_and_print_tests(self):
        """运行测试并打印所有结果数据"""
        print("=== 开始系统可用性测试 ===")
        print(f"测试服务器: {self.base_url}\n")
        
        # 运行所有测试
        health_test = self.test_health_check()
        workflow_tests = self.test_basic_workflow()
        
        # 合并所有测试结果
        all_tests = [health_test] + workflow_tests
        passed = sum(1 for t in all_tests if t.passed)
        
        # 打印每个测试的详细结果
        for test in all_tests:
            print(f"\n{'='*40}")
            print(f"测试名称: {test.name}")
            print(f"测试结果: {'通过' if test.passed else '失败'}")
            print(f"响应时间: {test.response_time:.2f}s")
            
            if test.error:
                print(f"\n错误信息: {test.error}")
            
            if test.response_data:
                print("\n请求负载:")
                pprint(test.response_data.get("request_payload"))
                
                print("\n响应数据:")
                pprint(test.response_data.get("data"))
        
        # 打印总结
        print(f"\n{'='*40}")
        print(f"=== 测试总结 ===")
        print(f"总测试数: {len(all_tests)}")
        print(f"通过数: {passed}")
        print(f"成功率: {passed/len(all_tests)*100:.1f}%")

if __name__ == "__main__":
    tester = AvailabilityTester()
    tester.run_and_print_tests()