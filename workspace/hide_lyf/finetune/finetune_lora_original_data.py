from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 加载tokenizer和教师模型
model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 确保pad_token设置正确

# 加载教师模型
teacher_model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 简化后的数据加载函数
def load_finetuning_data(data_path):
    all_data = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        # 处理列表格式数据
                        if isinstance(json_data, list):
                            for item in json_data:
                                if 'question' in item and 'analysis' in item:
                                    question = item['question']
                                    title = question.get('title', '')
                                    content = question.get('content', '')
                                    difficulty = question.get('difficulty', '')
                                    input_question = f"Title: {title}\nContent: {content}\nDifficulty: {difficulty}"
                                    
                                    all_data.append({
                                        "input": input_question,
                                        "output": item['analysis']
                                    })
                        # 处理字典格式数据
                        elif isinstance(json_data, dict):
                            if 'question' in json_data and 'analysis' in json_data:
                                question = json_data['question']
                                title = question.get('title', '')
                                content = question.get('content', '')
                                difficulty = question.get('difficulty', '')
                                input_question = f"Title: {title}\nContent: {content}\nDifficulty: {difficulty}"
                                
                                all_data.append({
                                    "input": input_question,
                                    "output": json_data['analysis']
                                })
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {file_path}")
    return all_data

# 加载数据
data_path = '../data/leetcode_data_multi'
train_data = load_finetuning_data(data_path)
batch_size = 8  # 可根据GPU内存调整

# 配置LoRA
lora_config = LoraConfig(
    r=8,  # 可尝试增大到16或32如果效果不佳
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 可添加"k_proj", "o_proj"
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA到教师模型
model = get_peft_model(teacher_model, lora_config)

# 定义优化器（添加weight decay防止过拟合）
optimizer = torch.optim.AdamW(model.parameters(), 
                            lr=2e-5,  # 适当提高学习率
                            weight_decay=0.01)
num_epochs = 10


# 批处理函数（简化版）
def process_batch(batch_data, tokenizer, max_length=1024):
    """
    处理batch数据，确保输入输出长度严格对齐
    返回: {
        "input_ids": tensor,
        "attention_mask": tensor,
        "labels": tensor
    }
    """
    # 构造完整文本 (输入+输出)
    texts = [f"问题: {data['input']}\n分析: {data['output']}" for data in batch_data]
    
    # 统一tokenize
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)
    
    # 创建labels (只对"分析:"后面的部分计算loss)
    labels = encodings.input_ids.clone()
    for i, text in enumerate(texts):
        # 找到"分析:"在tokenized结果中的位置
        analysis_pos = text.find("分析:")
        if analysis_pos == -1:
            labels[i, :] = -100  # 无效样本
            continue
            
        # 计算分析部分前的token数量
        prefix = text[:analysis_pos + len("分析:")]
        prefix_len = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
        
        # 设置忽略标记
        labels[i, :prefix_len] = -100  # 忽略输入部分
        # 忽略padding部分
        pad_mask = (encodings.input_ids[i] == tokenizer.pad_token_id)
        labels[i, pad_mask] = -100
    
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": labels
    }

# 训练循环（添加梯度裁剪）
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i + batch_size]

        try:
            # 处理batch数据
            batch = process_batch(batch_data, tokenizer)
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

            # 反向传播（添加梯度裁剪）
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:  # 每10个batch打印一次
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
            continue

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

# 保存模型（包含完整配置）
model.save_pretrained("lora-finetuned-teacher-model-original-data", 
                     safe_serialization=True)
tokenizer.save_pretrained("lora-finetuned-teacher-model-original-data")