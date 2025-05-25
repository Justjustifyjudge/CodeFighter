import torch
import json
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb  # 可选，如果需要记录到WandB

# ----------------------
# 配置参数
# ----------------------
# TEST_DATA_PATH = "test.json"
TRAINING_DATA_PATH_1 = "training_data_AT.json"
TRAINING_DATA_PATH_2 = "training_data_epoch3.json"
TRAINING_DATA_PATH_3 = "training_data_epoch10.json"
TRAINING_DATA_PATH_4 = "training_data_original_7B.json"
TEST_DATA_PATH=[TRAINING_DATA_PATH_1,TRAINING_DATA_PATH_2,TRAINING_DATA_PATH_3,TRAINING_DATA_PATH_4]
# TEST_DATA_PATH = "training_data_epoch3.json"

# MODEL_PATH = "dpo_scoring_model"  # 训练保存的模型路径
MODEL_PATH = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # 训练保存的模型路径
MAX_LENGTH = 1024
BATCH_SIZE = 4  # 根据GPU内存调整

# 初始化WandB（可选）
wandb.init(project="dpo_evaluation", name="test_evaluation")

# ----------------------
# 1. 加载测试数据
# ----------------------
print("Loading test data...")
def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)
def load_datas(data_paths):
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

test_dataset = load_data(TRAINING_DATA_PATH_1)
print(f"Loaded {len(test_dataset)} test samples")

# ----------------------
# 2. 加载训练好的模型和tokenizer
# ----------------------
print("Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# ----------------------
# 3. 定义评估函数
# ----------------------
def evaluate_model(model, tokenizer, dataset):
    results = []
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Evaluating"):
        batch = dataset[i:i+BATCH_SIZE]
        
        # Tokenization with consistent truncation/padding
        chosen_inputs = tokenizer(
            [p + c for p, c in zip(batch["prompt"], batch["chosen"])],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(model.device)
        
        rejected_inputs = tokenizer(
            [p + r for p, r in zip(batch["prompt"], batch["rejected"])],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(model.device)
        
        # 计算每个token的平均对数概率（关键修正）
        with torch.no_grad():
            # 计算chosen的logp
            chosen_loss = model(
                input_ids=chosen_inputs["input_ids"],
                attention_mask=chosen_inputs["attention_mask"],
                labels=chosen_inputs["input_ids"]
            ).loss
            chosen_logps = -chosen_loss * chosen_inputs["attention_mask"].sum(dim=1)  # 按实际长度计算
            
            # 计算rejected的logp
            rejected_loss = model(
                input_ids=rejected_inputs["input_ids"],
                attention_mask=rejected_inputs["attention_mask"],
                labels=rejected_inputs["input_ids"]
            ).loss
            rejected_logps = -rejected_loss * rejected_inputs["attention_mask"].sum(dim=1)
            
            # 转换为numpy数组
            chosen_logps = chosen_logps.cpu().numpy()
            rejected_logps = rejected_logps.cpu().numpy()
            
            # 数值检查
            assert not np.any(np.isnan(chosen_logps)), "NaN in chosen_logps"
            assert not np.any(np.isnan(rejected_logps)), "NaN in rejected_logps"
        
        # 收集结果
        for j in range(len(batch["prompt"])):
            margin = chosen_logps[j] - rejected_logps[j]
            results.append({
                "prompt": batch["prompt"][j],
                "chosen": batch["chosen"][j],
                "rejected": batch["rejected"][j],
                "chosen_logp": float(chosen_logps[j]),
                "rejected_logp": float(rejected_logps[j]),
                "margin": float(margin),
                "correct": margin > 0
            })
    
    # 计算指标
    accuracy = np.mean([r["correct"] for r in results]) if results else 0
    avg_margin = np.mean([r["margin"] for r in results]) if results else 0
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg Preference Margin: {avg_margin:.4f}")
    print(f"Sample Count: {len(results)}")
    
    # 记录到WandB
    wandb.log({
        "test_accuracy": accuracy,
        "test_preference_margin": avg_margin,
        "test_samples": len(results)
    })
    
    return {
        "accuracy": accuracy,
        "preference_margin": avg_margin,
        "total_samples": len(results),
        "detailed_results": results
    }

# ----------------------
# 4. 运行评估
# ----------------------
print("Starting evaluation...")
metrics = evaluate_model(model, tokenizer, test_dataset)

# 保存结果到JSON文件
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("Evaluation completed. Results saved to test_results.json")
wandb.finish()  # 结束WandB记录（如果启用）