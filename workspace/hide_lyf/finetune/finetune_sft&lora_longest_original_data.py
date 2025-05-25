from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os
import json
from tqdm import tqdm
from math import ceil

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

## longest需要防止OOM
torch.backends.cuda.enable_flash_sdp(True)  # 启用FlashAttention
torch.backends.cuda.enable_mem_efficient_sdp(True)  # 
# torch.backends.cuda.enable_math_sdp(True)

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
    # attn_implementation="sdpa"
)
teacher_model.gradient_checkpointing_enable()
# 数据加载函数保持不变
def load_finetuning_data(data_path):
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
batch_size = 2  # 可根据GPU内存调整

def process_batch(batch_data, tokenizer, max_total_length=4096, max_question_ratio=0.5, device=None):
    """
    动态调整输入输出比例，确保总长度不超过max_total_length
    max_question_ratio: 输入question最多占总长度的比例
    device: 指定设备
    """
    batch_inputs = []
    batch_outputs = []
    
    # 先单独tokenize输入和输出
    for data in batch_data:
        # Tokenize输入部分
        input_text = f"问题: {data['input']}\n分析: "
        input_tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_len = len(input_tokens.input_ids[0])
        
        # Tokenize输出部分
        output_tokens = tokenizer(data['output'], return_tensors="pt", add_special_tokens=False)
        output_len = len(output_tokens.input_ids[0])
        
        # 动态调整输入长度
        max_input_len = int(max_total_length * max_question_ratio)
        if input_len > max_input_len:
            input_tokens.input_ids = input_tokens.input_ids[:, :max_input_len]
            input_len = max_input_len
        
        # 计算允许的输出长度
        remaining_length = max_total_length - input_len - 3  # 留出特殊token空间
        if output_len > remaining_length:
            output_tokens.input_ids = output_tokens.input_ids[:, :remaining_length]
            output_len = remaining_length
        
        batch_inputs.append(input_tokens)
        batch_outputs.append(output_tokens)
    
    # 确定最大长度（考虑特殊token）
    max_len = max(
        len(input_tokens.input_ids[0]) + len(output_tokens.input_ids[0]) + 3 
        for input_tokens, output_tokens in zip(batch_inputs, batch_outputs)
    )
    
    # 构造完整序列 [BOS] input [SEP] output [EOS]
    full_ids = []
    for input_tokens, output_tokens in zip(batch_inputs, batch_outputs):
        input_part = input_tokens.input_ids[0]
        output_part = output_tokens.input_ids[0]
        
        # 构造序列并填充到最大长度
        seq = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            input_part,
            torch.tensor([tokenizer.sep_token_id]),
            output_part,
            torch.tensor([tokenizer.eos_token_id]),
            torch.full((max_len - len(input_part) - len(output_part) - 3,), 
                      tokenizer.pad_token_id)
        ])
        full_ids.append(seq)
    
    # 转换为张量
    full_ids = torch.stack(full_ids)
    
    # 创建attention mask和labels
    attention_mask = (full_ids != tokenizer.pad_token_id).long()
    labels = full_ids.clone()
    
    # 忽略输入部分的loss
    for i, (input_tokens, _) in enumerate(zip(batch_inputs, batch_outputs)):
        input_len = len(input_tokens.input_ids[0]) + 2  # +2 for BOS and SEP
        labels[i, :input_len] = -100
    
    # 移动到指定设备
    if device:
        full_ids = full_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 配置LoRA，longest要支持长序列
lora_config = LoraConfig(
    r=8,
    lora_alpha=32, # 增大alpha值
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 注意力层全覆盖
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True # 对长序列更稳定
)

num_epochs = 10

# 计算全参数微调和LoRA微调的step分界点
total_steps = ceil(len(train_data) / batch_size) * num_epochs
sft_steps = ceil(total_steps * 0.2)  # 使用20%的step进行全参数微调

# 训练循环
def train_hybrid_sft_lora(model, train_data, batch_size, num_epochs, sft_steps):
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=2e-5,
                                weight_decay=0.01)
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        if global_step >= sft_steps and not isinstance(model, PeftModel):
            print(f"Switching to LoRA training at step {global_step}")
            model.save_pretrained("full_sft_model_longest", safe_serialization=True)
            tokenizer.save_pretrained("full_sft_model_longest")
            print("Full SFT model saved")
            
            model = get_peft_model(model, lora_config)
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=2e-5,
                                        weight_decay=0.01)
            print("Converted to LoRA model")
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            
            try:
                batch = process_batch(
                    batch_data=batch_data,
                    tokenizer=tokenizer,
                    max_total_length=4096,
                    max_question_ratio=0.5,
                    device=model.device
                )
                
                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                global_step += 1
                
                if batch_count % 10 == 0:
                    mode = "Full SFT" if global_step < sft_steps else "LoRA"
                    print(f"Epoch {epoch+1}, Batch {batch_count} ({mode}), Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
                continue

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
    
    return model

# 开始训练
model = train_hybrid_sft_lora(teacher_model, train_data, batch_size, num_epochs, sft_steps)

# 保存最终模型（LoRA模型）
model.save_pretrained("hybrid_finetuned_model_longest", safe_serialization=True)
tokenizer.save_pretrained("hybrid_finetuned_model_longest")