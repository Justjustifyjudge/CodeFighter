from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
import json
from peft import PeftModel
from tqdm import tqdm
import wandb

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 初始化WandB
wandb.init(
    project="qwen2.5-coder-distillation-feature-based",  # 你的项目名称
    name="distillation-7B-from-32B",      # 本次运行的名称
    config={                              # 记录超参数配置
        "batch_size": 2,
        "num_epochs": 10,
        "learning_rate": 3e-5,
        "alpha": 0.7,
        "beta": 0.3,
        "temperature": 2.0,
        "accumulation_steps": 4,
        "max_length": 1024,
        "num_hidden_layers": 4
    }
)

config=wandb.config


# 模型加载
model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B'
tokenizer = AutoTokenizer.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# AT损失计算
class AttentionTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_atts, teacher_atts):
        loss = 0
        for s_att, t_att in zip(student_atts, teacher_atts):
            # 确保张量在同一设备上
            t_att = t_att.to(s_att.device)
            # 归一化注意力图
            s_att = torch.nn.functional.normalize(s_att, p=2, dim=-1)
            t_att = torch.nn.functional.normalize(t_att, p=2, dim=-1)
            # 计算MSE损失
            loss += torch.nn.functional.mse_loss(s_att, t_att)
        return loss / len(student_atts)

# 带隐藏状态捕获的模型包装器（适配Qwen2.5架构）
class QwenWithHiddenStates(torch.nn.Module):
    def __init__(self, base_model, num_hidden_layers=4):
        super().__init__()
        self.model = base_model
        self.hidden_states = []
        self.num_hidden_layers = num_hidden_layers
        self.hooks = []
        
        # 根据模型类型确定正确的层访问路径
        if isinstance(self.model, PeftModel):
            # PeftModel的路径: base_model.model.model.layers
            transformer_layers = self.model.base_model.model.model.layers
        else:
            # 普通模型的路径: model.model.layers
            transformer_layers = self.model.model.layers
        
        # 选择最后几层作为目标层
        target_layers = transformer_layers[-num_hidden_layers:]
        
        # 注册hook获取中间层输出
        for layer in target_layers:
            hook = layer.register_forward_hook(
                self._create_hook_fn()
            )
            self.hooks.append(hook)
    
    def _create_hook_fn(self):
        def hook(module, input, output):
            # Qwen2DecoderLayer的输出是一个元组，第一个元素是隐藏状态
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.hidden_states.append(hidden_state.detach())
        return hook
    
    def forward(self, **kwargs):
        self.hidden_states = []  # 清空之前的记录
        outputs = self.model(**kwargs)
        return outputs
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# 特征转换为注意力图
def features_to_attention(features):
    """将特征转换为注意力图"""
    # features形状: [batch_size, seq_len, hidden_dim]
    # 计算每个位置的特征范数作为注意力分数
    attention = torch.norm(features, p=2, dim=-1)  # [batch_size, seq_len]
    # 对序列维度进行softmax
    attention = torch.nn.functional.softmax(attention, dim=-1)
    return attention

# 初始化学生模型
student_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
student_model = QwenWithHiddenStates(student_model, config["num_hidden_layers"])

# 初始化教师模型
base_teacher_model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
teacher_model = PeftModel.from_pretrained(
    base_teacher_model,
    "/workspace/hide_lyf/finetune/lora-finetuned-teacher-model-code",
    device_map="auto"
)
teacher_model = QwenWithHiddenStates(teacher_model, config["num_hidden_layers"])

# 数据加载函数（保持不变）
def load_finetuning_data(data_path, model_type):
    all_data = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                        
                        for item in data:
                            if all(k in item for k in ['question', 'analysis', 'code']):
                                question = item['question']
                                input_question = f"Title: {question.get('title', '')}\nContent: {question.get('content', '')}"
                                difficulty = question.get('difficulty', '')
                                
                                # 处理代码占位符
                                analysis, placeholder_mapping = process_code_placeholders(item['analysis'])
                                
                                if difficulty == "Hard":
                                    code_str = "\n".join([f"{lang}代码:\n{code}" for lang, code in item['code'].items()])
                                    all_data.append({
                                        "input": input_question,
                                        "code_input": code_str,
                                        "output": analysis,
                                        "placeholder_mapping": placeholder_mapping
                                    })
                                else:
                                    code_output = "\n".join([f"```\n{code}\n```" for code in item['code'].values()])
                                    all_data.append({
                                        "input": input_question,
                                        "code_input": "",
                                        "output": code_output.strip(),
                                        "placeholder_mapping": {}
                                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    return all_data

def process_code_placeholders(text):
    placeholders = {}
    count = 0
    while True:
        start = text.find("```")
        if start == -1: break
        end = text.find("```", start+3)
        if end == -1: break
        code = text[start+3:end].strip()
        placeholder = f"[CODE_PLACEHOLDER_{count}]"
        text = text[:start] + placeholder + text[end+3:]
        placeholders[placeholder] = code
        count += 1
    return text, placeholders

# 修改后的损失函数
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, temperature=2.0):
        super().__init__()
        self.alpha = alpha  # Hard loss权重
        self.beta = beta    # AT loss权重
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.at_loss = AttentionTransferLoss()
    
    def forward(self, student_outputs, teacher_outputs, 
           student_hidden, teacher_hidden, labels):
        # 确保所有张量在同一设备上
        device = student_outputs.logits.device
        teacher_outputs.logits = teacher_outputs.logits.to(device)
        labels = labels.to(device)
        
        # Hard loss (交叉熵)
        hard_loss = self.ce_loss(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1)
        )
        
        # 将特征转换为注意力图
        student_atts = [features_to_attention(hid.to(device)) for hid in student_hidden]
        teacher_atts = [features_to_attention(hid.to(device)) for hid in teacher_hidden]
        
        # AT loss (注意力转移)
        at_loss = self.at_loss(student_atts, teacher_atts)
        
        # 总损失
        total_loss = self.alpha * hard_loss + self.beta * at_loss
        
        return total_loss, hard_loss, at_loss

# 数据处理函数（保持不变）
def prepare_batch(batch_data, tokenizer, max_length):
    inputs = tokenizer(
        [data["input"] for data in batch_data],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    labels = tokenizer(
        [restore_code_placeholders(data["output"], data["placeholder_mapping"]) 
         for data in batch_data],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).input_ids
    
    # 设置pad token的label为-100
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels

def restore_code_placeholders(text, mapping):
    for placeholder, code in mapping.items():
        text = text.replace(placeholder, f"```\n{code}\n```")
    return text

# 主训练流程
# 主训练流程（修改部分）
def train():
    # 加载数据
    train_data = load_finetuning_data("data/leetcode_problems", "thinking")
    
    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    total_steps = len(train_data) // config.batch_size * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = DistillationLoss(alpha=config.alpha, 
                               beta=config.beta,
                               temperature=config.temperature)
    
    # 训练循环
    for epoch in range(config.num_epochs):
        student_model.train()
        total_loss = 0
        total_hard = 0
        total_at = 0
        
        progress_bar = tqdm(range(0, len(train_data), config.batch_size), 
                         desc=f"Epoch {epoch+1}")
        
        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i:i+config.batch_size]
            
            # 准备数据
            inputs, labels = prepare_batch(batch, tokenizer, config.max_length)
            inputs = {k: v.to(student_model.model.device) for k, v in inputs.items()}
            labels = labels.to(student_model.model.device)
            
            # 教师模型前向
            with torch.no_grad():
                teacher_model.eval()
                teacher_outputs = teacher_model(**inputs)
                teacher_hidden = teacher_model.hidden_states[-config.num_hidden_layers:]
            
            # 学生模型前向
            student_model.train()
            student_outputs = student_model(**inputs)
            student_hidden = student_model.hidden_states[-config.num_hidden_layers:]
            
            # 计算损失
            loss, hard_loss, at_loss = criterion(
                student_outputs, teacher_outputs,
                student_hidden, teacher_hidden,
                labels
            )
            
            # 反向传播
            loss = loss / config.accumulation_steps
            loss.backward()
            
            if (i + 1) % config.accumulation_steps == 0 or (i + config.batch_size) >= len(train_data):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 记录统计信息
            total_loss += loss.item() * config.accumulation_steps
            total_hard += hard_loss.item()
            total_at += at_loss.item()
            
            # 记录到WandB
            wandb.log({
                "epoch": epoch + (i / len(train_data)),  # 连续记录epoch进度
                "loss": total_loss/(i//config.batch_size+1),
                "hard_loss": total_hard/(i//config.batch_size+1),
                "at_loss": total_at/(i//config.batch_size+1),
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            progress_bar.set_postfix({
                "loss": f"{total_loss/(i//config.batch_size+1):.4f}",
                "hard": f"{total_hard/(i//config.batch_size+1):.4f}",
                "AT": f"{total_at/(i//config.batch_size+1):.4f}"
            })
            progress_bar.update(config.batch_size)
        
        # 每个epoch结束后记录更多信息
        wandb.log({
            "epoch_loss": total_loss/(len(train_data)/config.batch_size),
            "epoch_hard_loss": total_hard/(len(train_data)/config.batch_size),
            "epoch_at_loss": total_at/(len(train_data)/config.batch_size)
        })
    
    # 保存检查点
    output_dir = f"distilled-model-epoch{config.num_epochs}"
    student_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 记录模型到WandB
    # wandb.save(os.path.join(output_dir, "*"))  # 保存所有模型文件
    
    # 训练完成后移除hook
    student_model.remove_hooks()
    teacher_model.remove_hooks()
    
    # 完成WandB运行
    wandb.finish()

if __name__ == "__main__":
    train()