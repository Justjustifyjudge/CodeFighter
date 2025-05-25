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


# 数据加载函数（修改版）
def load_finetuning_data(data_path, model_type):
    all_data = []
    code_placeholder_count = 0  # 记录占位符编号
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
                                    input_question = f"Title: {title}\nContent: {content}"

                                    code_str = ""
                                    code = item['code']
                                    # 处理analysis，添加占位符
                                    analysis = item['analysis']
                                    code_blocks = []  # 记录实际代码块
                                    placeholder_mapping = {}  # 记录占位符与代码块映射
                                    start_idx = 0
                                    while True:
                                        start = analysis.find("```", start_idx)
                                        if start == -1:
                                            break
                                        end = analysis.find("```", start + 3)
                                        code_block = analysis[start + 3: end].strip()
                                        placeholder = f"[CODE_PLACEHOLDER_{code_placeholder_count}]"
                                        analysis = analysis[:start] + placeholder + analysis[end + 3:]
                                        code_blocks.append(code_block)
                                        placeholder_mapping[placeholder] = code_block
                                        code_placeholder_count += 1
                                        start_idx = end + 3

                                    # 根据难度决定输出格式
                                    if difficulty == "Hard":
                                        # 困难题目: 保持原有格式(分析+代码)
                                        for lang, code_content in code.items():
                                            code_str += f"{lang} 代码:\n{code_content}\n"

                                        if model_type == "thinking":
                                            all_data.append({
                                                "input_1": input_question,
                                                "input_2": code_str,
                                                "output": analysis,
                                                "placeholder_mapping": placeholder_mapping  # 保存映射关系
                                            })
                                    else:
                                        # 非困难题目: 代码作为输出，用```分隔
                                        code_output = ""
                                        for lang, code_content in code.items():
                                            code_output += f"```\n{code_content}\n```\n"
                                        
                                        if model_type == "thinking":
                                            all_data.append({
                                                "input_1": input_question,
                                                "input_2": "",  # 非困难题目不需要额外代码输入
                                                "output": code_output.strip(),  # 纯代码输出
                                                "placeholder_mapping": {}  # 非困难题目不需要占位符映射
                                            })
                        elif isinstance(json_data, dict):
                            if 'question' in json_data and 'analysis' in json_data and 'code' in json_data:
                                question = json_data['question']
                                title = question.get('title', '')
                                content = question.get('content', '')
                                difficulty = question.get('difficulty', '')
                                input_question = f"Title: {title}\nContent: {content}"

                                code_str = ""
                                code = json_data['code']
                                # 处理analysis，添加占位符
                                analysis = json_data['analysis']
                                code_blocks = []
                                placeholder_mapping = {}
                                start_idx = 0
                                while True:
                                    start = analysis.find("```", start_idx)
                                    if start == -1:
                                        break
                                    end = analysis.find("```", start + 3)
                                    code_block = analysis[start + 3: end].strip()
                                    placeholder = f"[CODE_PLACEHOLDER_{code_placeholder_count}]"
                                    analysis = analysis[:start] + placeholder + analysis[end + 3:]
                                    code_blocks.append(code_block)
                                    placeholder_mapping[placeholder] = code_block
                                    code_placeholder_count += 1
                                    start_idx = end + 3

                                # 根据难度决定输出格式
                                if difficulty == "Hard":
                                    # 困难题目: 保持原有格式(分析+代码)
                                    for lang, code_content in code.items():
                                        code_str += f"{lang} 代码:\n{code_content}\n"

                                    if model_type == "thinking":
                                        all_data.append({
                                            "input_1": input_question,
                                            "input_2": code_str,
                                            "output": analysis,
                                            "placeholder_mapping": placeholder_mapping
                                        })
                                else:
                                    # 非困难题目: 代码作为输出，用```分隔
                                    code_output = ""
                                    for lang, code_content in code.items():
                                        code_output += f"```\n{code_content}\n```\n"
                                    
                                    if model_type == "thinking":
                                        all_data.append({
                                            "input_1": input_question,
                                            "input_2": "",  # 非困难题目不需要额外代码输入
                                            "output": code_output.strip(),  # 纯代码输出
                                            "placeholder_mapping": {}  # 非困难题目不需要占位符映射
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
        # 还原analysis中的代码（根据占位符映射）
        analysis = data["output"]
        placeholder_mapping = data["placeholder_mapping"]
        for placeholder, code_block in placeholder_mapping.items():
            analysis = analysis.replace(placeholder, code_block)

        # 构造输入部分文本
        input_text = f"问题:{data['input_1']}\n分析:"
        # 构造完整文本 (输入+输出)
        full_text = input_text + analysis
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
        batch_data = train_data[i:i + batch_size]

        try:
            # 使用新的批处理函数
            batch = process_batch(batch_data, tokenizer, max_length=1024)

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
            print(f"Epoch {epoch + 1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# 保存模型
model.save_pretrained("lora-finetuned-teacher-model-code")
tokenizer.save_pretrained("lora-finetuned-teacher-model-code")