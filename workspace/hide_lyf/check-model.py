from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
import json
from peft import PeftModel
from tqdm import tqdm
import inspect

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 超参数配置
config = {
    "batch_size": 1,  # 减小batch size防止OOM
    "num_epochs": 3,
    "learning_rate": 3e-5,
    "alpha": 0.7,
    "temperature": 2.0,
    "accumulation_steps": 4,
    "max_length": 1024,
    "num_hidden_layers": 4
}

# 模型加载
model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-7B'
tokenizer = AutoTokenizer.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Qwen2.5专用隐藏状态捕获器
class QwenHiddenStatesWrapper(torch.nn.Module):
    def __init__(self, base_model, num_hidden_layers=4):
        super().__init__()
        self.model = base_model
        self.hidden_states = []
        self.hooks = []
        
        # 动态获取Qwen2.5的transformer层
        transformer = self.model.model
        if hasattr(transformer, 'layers'):  # 标准结构
            layers = transformer.layers
        elif hasattr(transformer, 'transformer'):  # 某些变体
            layers = transformer.transformer.h
        else:
            # 打印模型结构以便调试
            print("Model structure inspection:")
            print(inspect.getmembers(transformer, lambda a: not inspect.isroutine(a)))
            raise ValueError("无法自动识别模型层结构，请手动指定")
        
        # 注册hook获取最后几层
        for layer in layers[-num_hidden_layers:]:
            hook = layer.register_forward_hook(
                lambda module, input, output: self.hidden_states.append(output[0].detach())
            )
            self.hooks.append(hook)
    
    def forward(self, **kwargs):
        self.hidden_states = []
        outputs = self.model(**kwargs)
        return outputs
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# 初始化模型
def init_model(model_path, is_teacher=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    
    if is_teacher:
        model = PeftModel.from_pretrained(
            model,
            "/workspace/hide_lyf/finetune/lora-finetuned-teacher-model-code",
            device_map="auto"
        )
    
    return QwenHiddenStatesWrapper(model, config["num_hidden_layers"])

# 初始化学生和教师模型
try:
    student_model = init_model(model_path)
    teacher_model = init_model(
        "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
        is_teacher=True
    )
except Exception as e:
    print(f"模型初始化失败: {str(e)}")
    # 打印教师模型结构以帮助调试
    temp_model = AutoModelForCausalLM.from_pretrained(
        "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B",
        trust_remote_code=True
    )
    print("\n教师模型结构:")
    print(temp_model)
    print("\nModel.model结构:")
    print(temp_model.model)
    raise

# [数据加载、损失函数等部分保持不变...]

# 测试中间层捕获
def test_hidden_states_capture():
    test_input = tokenizer("测试中间层捕获", return_tensors="pt").to(student_model.model.device)
    
    with torch.no_grad():
        _ = teacher_model(**test_input)
        print(f"教师模型捕获的隐藏层数: {len(teacher_model.hidden_states)}")
        for i, h in enumerate(teacher_model.hidden_states):
            print(f"层 {i+1} 形状: {h.shape}")
    
    _ = student_model(**test_input)
    print(f"学生模型捕获的隐藏层数: {len(student_model.hidden_states)}")
    for i, h in enumerate(student_model.hidden_states):
        print(f"层 {i+1} 形状: {h.shape}")

if __name__ == "__main__":
    test_hidden_states_capture()
    # train()  # 确保测试通过后再开始训练