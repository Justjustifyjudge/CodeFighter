import torch
import json
import os
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LongformerTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModel
)
from trl import DPOTrainer
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np
import random
import joblib
from torch import nn

# ----------------------
# 配置参数
# ----------------------
DATA_PATH = "../data/leetcode_problems"
DPO_OUTPUT_DIR = "dpo_scoring_model"
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# 1. 模型定义和全局变量
# ----------------------
classifier = None
label_encoder = None
classifier_tokenizer = None
generator_model = None
generator_tokenizer = None
reasoning_model = None
reasoning_tokenizer = None

# 分类器模型定义 (需与你训练的分类器结构一致)
class DifficultyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.longformer = AutoModel.from_pretrained('allenai/longformer-base-4096')
        self.out = nn.Linear(self.longformer.config.hidden_size, num_classes)  # 修改为out
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.out(pooled_output)  # 修改为out

# ----------------------
# 2. 模型加载函数 (与你提供的完全一致)
# ----------------------
def load_classifier():
    global classifier, label_encoder, classifier_tokenizer
    if classifier is None or label_encoder is None or classifier_tokenizer is None:
        print("Loading classifier components...")
        classifier_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        label_encoder = joblib.load('../classification/label_encoder.pkl')
        
        classifier = DifficultyClassifier(len(label_encoder.classes_)).to(DEVICE)
        classifier.load_state_dict(torch.load('../classification/best_model.bin', map_location=DEVICE))
        classifier.eval()
        print("Classifier components loaded successfully")

def load_generator():
    global generator_model, generator_tokenizer
    if generator_model is None or generator_tokenizer is None:
        print("Loading generator model and tokenizer...")
        model_path = "../distilled-model-epoch10"
        generator_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        generator_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
        print("Generator model and tokenizer loaded successfully")

def load_reasoning_model():
    global reasoning_model, reasoning_tokenizer
    if reasoning_model is None or reasoning_tokenizer is None:
        print("Loading reasoning model and tokenizer...")
        full_sft_model = AutoModelForCausalLM.from_pretrained(
            "../finetune/full_sft_model",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        reasoning_model = PeftModel.from_pretrained(
            full_sft_model,
            "../finetune/hybrid_finetuned_model"
        )
        
        reasoning_tokenizer = AutoTokenizer.from_pretrained(
            "../finetune/hybrid_finetuned_model",
            trust_remote_code=True
        )
        reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token
        print("Reasoning model and tokenizer loaded successfully")

# ----------------------
# 3. 辅助函数 - 模型调用
# ----------------------
def classify_difficulty(question_content):
    """使用分类模型判断题目难度"""
    load_classifier()  # 确保模型已加载
    
    inputs = classifier_tokenizer(
        question_content,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
        padding="max_length"
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = classifier(**inputs)
        pred_idx = torch.argmax(logits, dim=1).item()
    
    return label_encoder.inverse_transform([pred_idx])[0]

def generate_reasoning(question_content):
    """使用思考模型生成解题思路"""
    load_reasoning_model()  # 确保模型已加载
    
    prompt = f"请分析以下LeetCode题目并给出解题思路：\n{question_content}\n解题思路："
    inputs = reasoning_tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH).to(DEVICE)
    
    with torch.no_grad():
        outputs = reasoning_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=reasoning_tokenizer.eos_token_id
        )
    
    reasoning = reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reasoning.replace(prompt, "").strip()

def generate_code(question_content, reasoning=None):
    """使用生成模型生成代码"""
    load_generator()  # 确保模型已加载
    
    if reasoning:
        prompt = (
            f"LeetCode题目：\n{question_content}\n"
            f"解题思路：\n{reasoning}\n\n"
            "请根据上述思路编写Python代码：\n```python\n"
        )
    else:
        prompt = f"请为以下LeetCode题目生成Python代码：\n{question_content}\n```python\n"
    
    inputs = generator_tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH).to(DEVICE)
    
    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            pad_token_id=generator_tokenizer.eos_token_id
        )
    
    full_output = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取代码部分
    if "```python" in full_output:
        code = full_output.split("```python")[1].split("```")[0].strip()
    elif "```" in full_output:
        code = full_output.split("```")[1].split("```")[0].strip()
    else:
        lines = full_output.split("\n")
        code_lines = [line for line in lines if line.strip().startswith(("def ", "class ", "import ")) or line.startswith("    ")]
        code = "\n".join(code_lines).strip() if code_lines else full_output
    
    return code

# ----------------------
# 4. 改进的rejected代码生成
# ----------------------
def generate_rejected_code(question):
    """完整的rejected代码生成流程"""
    content = question["content"]
    
    # 第一步：难度分类
    difficulty = classify_difficulty(content)
    
    # 第二步：如果是难题，生成思考
    reasoning = None
    if difficulty == "hard":
        reasoning = generate_reasoning(content)
    
    # 第三步：生成代码
    return generate_code(content, reasoning)

# ----------------------
# 5. 数据加载和预处理 (保持不变)
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
                                if isinstance(data["code"], dict):
                                    code_values = list(data["code"].values())
                                    if len(code_values) > 0:
                                        code = code_values[-1]
                                    else:
                                        code = ""
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
    random.shuffle(raw_data)
    return raw_data

def is_valid_data(item):
    """检查数据是否合法"""
    question = item["question"]
    chosen_code = item["code"]
    if not isinstance(question, dict) or 'content' not in question:
        return False
    if not chosen_code.strip():
        return False
    return True

# ----------------------
# 6. 构建偏好数据集
# ----------------------
def build_preference_dataset(raw_data, num_samples=500):
    """
    构建偏好数据集：
    - chosen: 原始数据中的真实代码
    - rejected: 通过完整流程生成的代码
    """
    # 预加载所有模型
    load_classifier()
    load_generator()
    load_reasoning_model()
    
    dataset = []
    for item in tqdm(raw_data[:num_samples], desc="Building DPO dataset"):
        if not is_valid_data(item):
            continue
            
        question = item["question"]
        chosen_code = item["code"]

        # 生成rejected代码（使用完整流程）
        rejected_code = generate_rejected_code(question)

        # 确保两者不同且rejected代码不为空
        if chosen_code.strip() != rejected_code.strip() and rejected_code.strip():
            dataset.append({
                "prompt": question["content"],
                "chosen": chosen_code,
                "rejected": rejected_code,
                "difficulty": classify_difficulty(question["content"])  # 记录难度
            })

    print(f"Final dataset size: {len(dataset)}")
    return Dataset.from_list(dataset)

# ----------------------
# 主执行流程
# ----------------------
if __name__ == "__main__":
    print("Loading raw data...")
    raw_data = load_raw_data(DATA_PATH)
    
    print("\nBuilding preference dataset...")
    preference_dataset = build_preference_dataset(raw_data, num_samples=300)
    
    # 保存数据集
    training_data_path = "training_data_AT.json"
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(preference_dataset.to_list(), f, ensure_ascii=False, indent=4)
    print(f"\nTraining data saved to {training_data_path}")
    
    # 打印示例
    print("\nSample data:")
    print(json.dumps(preference_dataset[0], indent=2, ensure_ascii=False))