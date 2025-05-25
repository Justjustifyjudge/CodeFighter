####################### 模型定义
from transformers import AutoModelForCausalLM, AutoTokenizer#, Qwen2Tokenizer
import torch
import torch.nn as nn
import os
import json
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B'
tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B", trust_remote_code=True)
thinking_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).eval()
# speaking_model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# ).eval()

######################## 蒸馏过程

# 加载教师模型
# teacher_code_model = AutoModelForCausalLM.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B-Instruct").to(speaking_model.device)
base_teacher_model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    device_map="auto",  # 指定加载到 CPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).eval()

teacher_think_model = PeftModel.from_pretrained(
    base_teacher_model,
    "/workspace/hide_lyf/finetune/lora-finetuned-teacher-model-code",
    device_map="auto"
).eval()

# 定义损失函数
# def combined_kl_loss(student_logits, teacher_logits, alpha=0.5):
#     kl_loss = nn.KLDivLoss(reduction="batchmean")
#     forward_kl = kl_loss(torch.log_softmax(student_logits, dim=-1), torch.softmax(teacher_logits, dim=-1))
#     backward_kl = kl_loss(torch.log_softmax(teacher_logits, dim=-1), torch.softmax(student_logits, dim=-1))

#     return alpha * forward_kl + (1 - alpha) * backward_kl
def combined_kl_loss(student_logits, teacher_logits, alpha=0.5, temperature=1.0):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    soft_student = torch.softmax(student_logits / temperature, dim=-1)
    
    forward_kl = kl_loss(torch.log(soft_student), soft_teacher)
    backward_kl = kl_loss(torch.log(soft_teacher), soft_student)
    
    return alpha * forward_kl + (1 - alpha) * backward_kl

############# 数据加载
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

data_path='data/leetcode_problems'
thinking_sft_data = load_finetuning_data(data_path, model_type="thinking")

batch_size = 2  # 你可以根据 GPU 内存和数据量调整这个值

# # 训练循环（以Thinking Model为例，Speaking Model同理）
optimizer = torch.optim.Adam(thinking_model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    for i in range(0, len(thinking_sft_data), batch_size):
        batch_data=thinking_sft_data[i:i+batch_size]
        input_texts = [data["input_1"] for data in batch_data]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            teacher_outputs = teacher_think_model(**inputs)
            teacher_logits = teacher_outputs.logits

        student_outputs = thinking_model(**inputs)
        student_logits = student_outputs.logits

        loss = combined_kl_loss(student_logits, teacher_logits, alpha=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"处理完当前epoch {i} 条json")

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

# ################# 模型保存

# # 保存speaking Model
thinking_model.save_pretrained("distilled-speaking-model")
tokenizer.save_pretrained("distilled-speaking-model")