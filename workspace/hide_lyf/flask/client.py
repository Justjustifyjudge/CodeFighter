import requests
import json
import sys

# 服务器地址
SERVER_URL = "http://localhost:5001/generate_code"  # 根据实际情况修改

def get_multiline_input(prompt):
    """获取多行输入，直到遇到空行"""
    print(prompt)
    print("(输入完问题后，请连续按两次回车确认)")
    lines = []
    while True:
        line = input()
        if line.strip() == '':
            if lines:  # 已经有内容时才结束
                break
            else:
                print("问题不能为空，请继续输入")
                continue
        lines.append(line)
    return '\n'.join(lines)

def call_api(problem_text, difficulty=None):
    """调用生成代码的API"""
    payload = {
        "problem": problem_text,
        "temperature": 0.7,
        "max_length": 1024
    }
    
    if difficulty:
        payload["difficulty"] = difficulty
    
    try:
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"\nAPI请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"\n请求出错: {str(e)}")
        return None

def main():
    print("多行代码生成客户端 (输入'quit'退出)")
    print("--------------------------------")
    print("提示：可以输入多行问题，输入完成后连续按两次回车确认")
    
    while True:
        # 读取多行用户输入
        problem = get_multiline_input("\n请输入问题描述:")
        
        if problem.lower().strip() in ('quit', 'exit', 'q'):
            break
            
        # 可选: 询问是否指定难度级别
        difficulty = None
        set_difficulty = input("\n是否指定难度级别? (y/n, 默认为n): ").strip().lower()
        if set_difficulty == 'y':
            while True:
                difficulty = input("请输入难度级别 (Easy/Medium/Hard): ").strip().capitalize()
                if difficulty in ('Easy', 'Medium', 'Hard'):
                    break
                print("无效的难度级别，请重新输入")
        
        # 调用API
        print("\n正在生成代码，请稍候...")
        result = call_api(problem, difficulty)
        
        # 显示结果
        if result:
            print("\n生成结果:")
            print(f"问题难度: {result.get('difficulty', '未知')}")
            
            if 'reasoning' in result:
                print("\n问题分析:")
                print(result['reasoning'])
                
            print("\n生成的代码:")
            print(result.get('result', '无结果'))
            
            print("\n详细信息:")
            print(f"- 是否显式指定难度: {'是' if result.get('is_explicit_difficulty', False) else '否'}")
            print(f"- 输入长度: {result.get('input_length', 0)} 字符")
            print(f"- 输出长度: {result.get('output_length', 0)} 字符")
        else:
            print("\n生成代码失败")

if __name__ == "__main__":
    main()