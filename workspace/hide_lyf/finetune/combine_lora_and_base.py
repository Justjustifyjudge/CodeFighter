from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

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
lora_model_path = "distilled-teacher-lora-model"
model = PeftModel.from_pretrained(base_model, lora_model_path)

# 合并 LoRA 权重到基础模型
model = model.merge_and_unload()

# 保存合并后的模型
merged_model_path = "merged-distilled-teacher-model"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

# 重新加载合并后的模型
reloaded_model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 使用重新加载的模型进行推理
input_text = "这里输入你的测试文本"
inputs = tokenizer(input_text, return_tensors="pt").to(reloaded_model.device)
outputs = reloaded_model.generate(**inputs)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)