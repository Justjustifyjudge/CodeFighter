# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class DPOCodeScorer:
    def __init__(self, model_path, device="auto", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,  # 保持与训练时一致的精度
            trust_remote_code=True
        ).eval()
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 确保pad_token设置

    def _get_logits(self, text):
        """获取模型对输入的logits（核心打分逻辑）"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 取最后一个token的logits作为分数依据
            return outputs.logits[:, -1, :].mean().item()

    def score(self, question, code):
        """获取代码的偏好分数（越高表示越偏好）"""
        # 构造DPO训练时的prompt格式（需与训练数据格式一致！）
        text = f"Question: {question}\n\nCode: {code}\n\nLabel:"
        return self._get_logits(text)

    def compare(self, question, code_a, code_b):
        """比较两个代码的优劣"""
        score_a = self.score(question, code_a)
        score_b = self.score(question, code_b)
        return {
            "preferred": "A" if score_a > score_b else "B",
            "score_a": score_a,
            "score_b": score_b,
            "explanation": f"DPO模型认为代码{'A' if score_a > score_b else 'B'}更符合偏好"
        }

# 示例使用
if __name__ == "__main__":
    DPO_MODEL_PATH = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # 替换为你的模型路径
    
    scorer = DPOCodeScorer(DPO_MODEL_PATH)
    
    question = "Write a function to reverse a linked list."
    code_a = """
    def reverse_list(head):
        prev = None
        while head:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev
    """
    code_b = """
    def reverse(head):
        return head[::-1]
    """
    
    result = scorer.compare(question, code_a, code_b)
    print("Comparison Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))