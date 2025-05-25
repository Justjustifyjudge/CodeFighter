from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
from peft import PeftModel, LoraConfig
import torch

# 生成配置
generation_config = GenerationConfig(
    max_new_tokens=4096,  # 设置为需要的最大长度
    min_new_tokens=100,
    do_sample=True,
    temperature=0.8,      # 更高温度增加多样性
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.15,
    length_penalty=1.0,    # 避免过早结束
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True        # 必须开启以支持长序列
)

# 保存配置
generation_config.save_pretrained("hybrid_finetuned_model")

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
    
    # 使用流式生成避免内存爆炸
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        streamer=streamer,
        max_new_tokens=max_length
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output

# 使用示例
input_question = "如何用Python实现快速排序？"
print(generate_response(hybrid_model, tokenizer, input_question))