from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

# 加载基础模型
base_model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载 LoRA 模型
lora_model_path = "lora-finetuned-teacher-model-original-data"
model = PeftModel.from_pretrained(base_model, lora_model_path)

# 使用加载的模型进行推理
input_text = "问题：计算两数之和，提供a、b两个整数，返回它们的和。"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)