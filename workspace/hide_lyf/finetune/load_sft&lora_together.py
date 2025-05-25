from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

# 1. 先加载全参数微调模型
full_sft_model = AutoModelForCausalLM.from_pretrained(
    "full_sft_model",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 2. 在全量模型基础上加载LoRA
hybrid_model = PeftModel.from_pretrained(
    full_sft_model,
    "hybrid_finetuned_model"
)

# # 3. 如需永久合并（可选）
# final_model = hybrid_model.merge_and_unload()

# 加载我自己的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "hybrid_finetuned_model",  # 您保存的路径
    trust_remote_code=True
)

# 确保pad_token设置正确（与训练时一致）
tokenizer.pad_token = tokenizer.eos_token

def generate_response(model, tokenizer, input_text):
    inputs = tokenizer(f"问题: {input_text}\n分析:", return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用示例
input_question = "如何用Python实现快速排序？"
print(generate_response(hybrid_model, tokenizer, input_question))