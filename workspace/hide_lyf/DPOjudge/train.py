import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ----------------------
# 配置参数
# ----------------------
TRAINING_DATA_PATH_1 = "training_data_AT.json"
TRAINING_DATA_PATH_2 = "training_data_epoch3.json"
TRAINING_DATA_PATH_3 = "training_data_epoch10.json"
TRAINING_DATA_PATH_4 = "training_data_original_7B.json"
DPO_MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
DPO_OUTPUT_DIR = "dpo_scoring_model"
MAX_LENGTH = 1024
TRAINING_DATA_PATH=[TRAINING_DATA_PATH_1,TRAINING_DATA_PATH_2,TRAINING_DATA_PATH_3,TRAINING_DATA_PATH_4]
# ----------------------
# 1. 加载训练数据
# ----------------------
print("Loading training data...")
def load_training_data(data_paths):
    all_data = []
    for data_path in data_paths:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_data.extend(data)
        except FileNotFoundError:
            print(f"错误：文件 {data_path} 未找到。")
        except json.JSONDecodeError:
            print(f"错误：无法解析 {data_path} 为有效的 JSON。")
    return Dataset.from_list(all_data)

preference_dataset = load_training_data(TRAINING_DATA_PATH)
print(f"Loaded {len(preference_dataset)} training samples")

# ----------------------
# 2. 初始化模型和tokenizer
# ----------------------
print("Initializing DPO model...")
tokenizer = AutoTokenizer.from_pretrained(DPO_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    DPO_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ----------------------
# 3. 配置并运行DPO训练
# ----------------------
dpo_config = DPOConfig(
    output_dir=DPO_OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=20,
    bf16=True,
    remove_unused_columns=False,
    optim="adamw_torch",
    max_length=MAX_LENGTH,
    max_prompt_length=1024,
    beta=0.1  # DPO强度参数
)

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

print("Starting DPO training...")
dpo_trainer.train()
dpo_trainer.save_model(DPO_OUTPUT_DIR)
tokenizer.save_pretrained(DPO_OUTPUT_DIR)

print("\nTraining completed and model saved.")