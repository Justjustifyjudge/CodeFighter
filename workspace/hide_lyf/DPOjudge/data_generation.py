import torch
import json
import os
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model
import numpy as np
import random

# ----------------------
# 配置参数
# ----------------------
STUDENT_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B"  # 你的蒸馏后模型路径
DATA_PATH = "../data/leetcode_problems"              # 原始数据路径
MAX_LENGTH = 1024                                # 最大输入长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# 1. 加载生成模型（用于生成rejected样本）
# ----------------------
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

# 加载蒸馏后的学生模型（用于生成rejected样本）
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# ----------------------
# 2. 构建偏好数据集（直接使用原始数据中的code字段作为chosen）
# ----------------------
def load_raw_data(data_path):
    """加载原始数据，提取问题和真实代码"""
    raw_data = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'question' in item and 'code' in item:
                                    # 若code是列表，取最后一个元素
                                    if isinstance(item["code"], list) and len(item["code"]) > 0:
                                        code = item["code"][-1]
                                    else:
                                        code = item["code"]
                                    raw_data.append({
                                        "question": item["question"],
                                        "code": code
                                    })
                        elif isinstance(data, dict):
                            if 'question' in data and 'code' in data:
                                # 若code是字典，取最后一个值
                                if isinstance(data["code"], dict):
                                    code_values = list(data["code"].values())
                                    if len(code_values) > 0:
                                        code = code_values[-1]
                                    else:
                                        code = ""
                                # 若code是列表，取最后一个元素
                                elif isinstance(data["code"], list) and len(data["code"]) > 0:
                                    code = data["code"][-1]
                                else:
                                    code = data["code"]
                                raw_data.append({
                                    "question": data["question"],
                                    "code": code
                                })
                    except json.JSONDecodeError:
                        print(f"Error loading {file}")
    # 打乱数据
    random.shuffle(raw_data)
    return raw_data

def generate_rejected_code(question, model, tokenizer):
    """用学生模型生成rejected代码"""
    prompt = f"{question['content']}\n生成代码:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.2,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 改进的代码提取逻辑
    # 情况1：有代码块标记
    if "```python" in generated:
        parts = generated.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0].strip()
            return code_part
    
    # 情况2：有通用代码块标记
    elif "```" in generated:
        parts = generated.split("```")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0].strip()
            # 去除可能的语言标识符
            if code_part.startswith("python\n"):
                code_part = code_part[7:]
            return code_part
    
    # 情况3：有"生成代码:"标记
    elif "生成代码:" in generated:
        parts = generated.split("生成代码:")
        if len(parts) > 1:
            code_part = parts[1].strip()
            # 去除可能的后续指令
            end_markers = ["\n生成代码:", "\nAnswer:", "\n```", "\n问题:"]
            for marker in end_markers:
                if marker in code_part:
                    code_part = code_part.split(marker)[0]
            return code_part
    
    # 情况4：没有明显标记，尝试提取第一段代码结构
    lines = generated.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        # 检查行是否像代码（包含Python关键字或缩进）
        if (line.strip().startswith(("def ", "class ", "import ", "from ")) or 
            (line.startswith("    ") and not line.strip().startswith("#"))):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return "\n".join(code_lines).strip()
    
    # 作为最后手段，返回整个生成内容
    return generated.strip()

def is_valid_data(item):
    """检查数据是否合法"""
    question = item["question"]
    chosen_code = item["code"]
    if not isinstance(question, dict) or 'content' not in question:
        return False
    if not chosen_code.strip():
        return False
    return True

def build_preference_dataset(raw_data, num_samples=500):
    """
    构建偏好数据集：
    - chosen: 原始数据中的真实代码
    - rejected: 学生模型生成的代码
    """
    dataset = []
    for item in tqdm(raw_data[:num_samples], desc="Building dataset"):
        if not is_valid_data(item):
            continue
        question = item["question"]
        chosen_code = item["code"]

        # 生成rejected代码
        rejected_code = generate_rejected_code(question, student_model, tokenizer)

        # 确保两者不同且rejected代码不为空
        if chosen_code.strip() != rejected_code.strip() and rejected_code.strip():
            dataset.append({
                "prompt": question["content"],  # 使用prompt作为标准字段
                "chosen": chosen_code,
                "rejected": rejected_code
            })

    print(f"Final dataset size: {len(dataset)} (Chosen=原始代码, Rejected=模型生成)")
    return Dataset.from_list(dataset)

# 加载数据并构建数据集
print("Loading raw data...")
raw_data = load_raw_data(DATA_PATH)
preference_dataset = build_preference_dataset(raw_data, num_samples=300)  # 示例用300条数据

# 保存训练数据到JSON文件
training_data_path = "training_data_original_7B.json"
with open(training_data_path, 'w', encoding='utf-8') as f:
    json.dump(preference_dataset.to_list(), f, ensure_ascii=False, indent=4)
print(f"Training data saved to {training_data_path}")