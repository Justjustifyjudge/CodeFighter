import torch
from transformers import LongformerTokenizer
import joblib
from model import DifficultyClassifier
from dataloader import create_data_loader, DifficultyDataset

class LongformerDifficultyPredictor:
    def __init__(self, model_path, label_encoder_path, max_length=1024):
        """初始化预测器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # 加载所需组件
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.label_encoder = joblib.load(label_encoder_path)
        
        # 加载模型 (需提前定义好DifficultyClassifier类)
        self.model = DifficultyClassifier(len(self.label_encoder.classes_)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, text):
        """预测单个文本的难度标签"""
        # 文本编码
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 预测
        with torch.no_grad():
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            pred_idx = torch.argmax(outputs).item()
        
        # 返回原始标签
        return self.label_encoder.inverse_transform([pred_idx])[0]

# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = LongformerDifficultyPredictor(
        model_path='best_model.bin',
        label_encoder_path='label_encoder.pkl'
    )
    
    # 预测单个文本
    text = "给定一个整数数组，找出所有元素的和"
    print(f"预测结果: {predictor.predict(text)}")