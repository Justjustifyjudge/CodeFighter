from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AdaptionPromptConfig, get_peft_model, AdapterConfig  # 改为使用Adapter
import torch
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 加载tokenizer和教师模型
model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载教师模型
teacher_model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 数据加载函数（保持不变）
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
batch_size = 8

# 配置Adapter（会报错，Unsupported model type for adaption prompt: '{model.config.model_type}'.说明不支持Qwen2.5模型架构）
# adapter_config = AdaptionPromptConfig(
#     adapter_len=10,       # Adapter长度
#     adapter_layers=30,    # 插入Adapter的层数
#     task_type="CAUSAL_LM"
# )
adapter_config = AdapterConfig(
    mh_adapter=True,      #注意力层添加Adapter
    output_adapter=True,  # 输出层添加Adapter
    reduction_facter=24,   # 瓶颈层缩减因子，平衡效果和参数量
    non_linearity="gelu",  #激活函数
    init_weights="bert", #初始化方式
    task_type="CAUSAL_LM"
)


# 应用Adapter到教师模型
model = get_peft_model(teacher_model, adapter_config)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 3

# 批处理函数（保持不变）
def process_batch(batch_data, tokenizer, max_length=1024):
    texts = [f"问题: {data['input']}\n分析: {data['output']}" for data in batch_data]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)
    
    labels = encodings.input_ids.clone()
    for i, text in enumerate(texts):
        analysis_pos = text.find("分析:")
        if analysis_pos == -1:
            labels[i, :] = -100
            continue
            
        prefix = text[:analysis_pos + len("分析:")]
        prefix_len = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
        labels[i, :prefix_len] = -100
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
            batch = process_batch(batch_data, tokenizer)
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
            continue

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

# 保存模型
model.save_pretrained("adapter-finetuned-teacher-model")
tokenizer.save_pretrained("adapter-finetuned-teacher-model")