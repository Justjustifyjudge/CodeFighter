from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 加载tokenizer和教师模型
model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 确保pad_token设置正确

# 加载教师模型
teacher_model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 数据加载函数（保持不变）
def load_finetuning_data(data_path, model_type):
    all_data = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            for item in json_data:
                                if 'question' in item and 'analysis' in item and 'code' in item:
                                    question = item['question']
                                    title = question.get('title', '')
                                    content = question.get('content', '')
                                    difficulty = question.get('difficulty', '')
                                    input_question = f"Title: {title}\nContent: {content}\nDifficulty: {difficulty}"

                                    code_str = ""
                                    code = item['code']
                                    for lang, code_content in code.items():
                                        code_str += f"{lang} 代码:\n{code_content}\n"

                                    if model_type == "thinking":
                                        all_data.append({
                                            "input_1": input_question,
                                            "input_2": code_str,
                                            "output": item['analysis']
                                        })
                        elif isinstance(json_data, dict):
                            if 'question' in json_data and 'analysis' in json_data and 'code' in json_data:
                                question = json_data['question']
                                title = question.get('title', '')
                                content = question.get('content', '')
                                difficulty = question.get('difficulty', '')
                                input_question = f"Title: {title}\nContent: {content}\nDifficulty: {difficulty}"

                                code_str = ""
                                code = json_data['code']
                                for lang, code_content in code.items():
                                    code_str += f"{lang} 代码:\n{code_content}\n"

                                if model_type == "thinking":
                                    all_data.append({
                                        "input_1": input_question,
                                        "input_2": code_str,
                                        "output": json_data['analysis']
                                    })
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {file_path}")
    return all_data

# 加载数据
data_path = '../data/leetcode_problems'
train_data = load_finetuning_data(data_path, model_type="thinking")
batch_size = 8

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA到教师模型
model = get_peft_model(teacher_model, lora_config)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 3

# 关键修改：实现强制长度对齐的批处理函数
def process_batch(batch_data, tokenizer, max_length=512):
    """
    处理batch数据，确保输入输出长度严格对齐
    返回: {
        "input_ids": tensor,
        "attention_mask": tensor,
        "labels": tensor
    }
    """
    # 构造完整文本 (输入+输出)
    full_texts = []
    input_lengths = []
    for data in batch_data:
        # 构造输入部分文本
        input_text = f"问题:{data['input_1']}\n分析:"
        # 构造完整文本 (输入+输出)
        full_text = input_text + data['output']
        full_texts.append(full_text)
        
        # 单独tokenize输入部分获取其长度
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids[0]
        input_lengths.append(len(input_ids))
    
    # 统一tokenize完整文本
    encodings = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)
    
    # 创建labels (只对输出部分计算loss)
    labels = encodings.input_ids.clone()
    for i, input_len in enumerate(input_lengths):
        # 将输入部分(问题+分析提示)的label设置为-100(忽略)
        labels[i, :input_len] = -100
        
        # 如果输出部分被截断，也忽略被截断的部分
        if input_len >= max_length:
            labels[i, :] = -100  # 整个样本无效
        else:
            # 确保没有被padding的部分也被忽略
            pad_mask = (encodings.input_ids[i] == tokenizer.pad_token_id)
            labels[i, pad_mask] = -100
    
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": labels
    }

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        
        try:
            # 使用新的批处理函数
            batch = process_batch(batch_data, tokenizer)
            
            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# 保存模型
model.save_pretrained("lora-finetuned-teacher-model")
tokenizer.save_pretrained("lora-finetuned-teacher-model")