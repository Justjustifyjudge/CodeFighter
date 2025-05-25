import torch
from transformers import LongformerTokenizer
from torch import nn
from transformers import LongformerModel
from dataloader import DifficultyDataset
from model import DifficultyClassifier

def load_trained_model(model_path, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 初始化模型结构
    model = DifficultyClassifier(n_classes=num_classes)
    model.to(device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    
    return model

# 使用示例
if __name__ == "__main__":
    # 参数设置
    MODEL_PATH = 'best_model.bin'  # 或 'difficulty_classifier_final.bin'
    NUM_CLASSES = 3  # 替换为你的实际类别数
    MAX_LENGTH = 1024
    TEST_TEXT = "这是一个测试问题内容..."

    # 加载模型和tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = load_trained_model(MODEL_PATH, NUM_CLASSES, device)

    # 预处理输入文本
    encoded_text = tokenizer.encode_plus(
        TEST_TEXT,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 预测
    with torch.no_grad():
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        print(f"Predicted class index: {prediction.item()}")