import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

class DifficultyDataset(Dataset):
    def __init__(self, json_folder, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.contents = []
        self.labels = []
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        
        # 收集所有数据
        all_difficulties = []
        for json_file in os.listdir(json_folder):
            if json_file.endswith('.json'):
                with open(os.path.join(json_folder, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question = data['question']
                    self.contents.append(question['content'])
                    all_difficulties.append(question['difficulty'])
        
        # 编码标签
        self.labels = self.label_encoder.fit_transform(all_difficulties)
        self.num_classes = len(self.label_encoder.classes_)

        # 编码标签后，保存label_encoder
        self.labels = self.label_encoder.fit_transform(all_difficulties)
        self.num_classes = len(self.label_encoder.classes_)
        
        # 保存label_encoder到文件
        import joblib
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
    
    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, idx):
        content = self.contents[idx]
        label = self.labels[idx]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(json_folder, tokenizer, batch_size=16, max_length=1024):
    dataset = DifficultyDataset(json_folder, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def check_max_tokens(json_folder, tokenizer):
    max_tokens = 0
    max_content = ""
    
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = data['question']['content']
                
                # 使用 tokenizer 编码文本（不填充、不截断）
                tokens = tokenizer.encode(content, add_special_tokens=True)
                token_count = len(tokens)
                
                # 更新最大值
                if token_count > max_tokens:
                    max_tokens = token_count
                    max_content = content[:50]  # 只保存前50字符用于展示
    
    print(f"最大 token 数量: {max_tokens}")
    print(f"对应的文本片段: {max_content}...")
    return max_tokens

if __name__ == "__main__":
    # # 初始化 tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # # 设置 JSON 文件夹路径
    # pth1 = "/workspace/hide_lyf/data/leetcode_data_multi"  # 替换为实际路径
    
    # # 检查最大 token 数量
    # max_tokens = check_max_tokens(pth1, tokenizer)
    
    # # 根据检查结果建议 max_length
    # print(f"\n建议的 max_length 设置:")
    # print(f"- 最低安全值: {max_tokens} (覆盖最长文本)")
    # print(f"- 平衡值: {min(max_tokens + 10, 512)} (稍加缓冲，但不超过 BERT 的 512 限制)")
    # print(f"- 保守值: 128 (适用于短文本)")
    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 设置 JSON 文件夹路径
    pth1 = "/workspace/hide_lyf/data/leetcode_data_multi"  # 替换为实际路径
    
    # 1. 检查最大 token 数量
    print("=== Token 长度分析 ===")
    max_tokens = check_max_tokens(pth1, tokenizer)
    
    # 根据检查结果建议 max_length
    print(f"\n建议的 max_length 设置:")
    print(f"- 最低安全值: {max_tokens} (覆盖最长文本)")
    print(f"- 平衡值: {min(max_tokens + 10, 512)} (稍加缓冲，但不超过 BERT 的 512 限制)")
    print(f"- 保守值: 128 (适用于短文本)")
    
    # 2. 收集所有数据用于统计分析
    print("\n=== 数据分布分析 ===")
    all_contents = []
    all_difficulties = []
    content_lengths = []
    
    for json_file in os.listdir(pth1):
        if json_file.endswith('.json'):
            with open(os.path.join(pth1, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = data['question']
                content = question['content']
                difficulty = question['difficulty']
                
                all_contents.append(content)
                all_difficulties.append(difficulty)
                content_lengths.append(len(content))
    
    # 3. 难度等级分布分析
    print("\n=== 难度等级分布 ===")
    difficulty_counts = {}
    for d in all_difficulties:
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    # 打印分布情况
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"{difficulty}: {count} 样本 ({count/len(all_difficulties):.2%})")
    
    # 4. 文本长度分析
    print("\n=== 文本长度分析 ===")
    print(f"总样本数: {len(content_lengths)}")
    print(f"平均长度: {sum(content_lengths)/len(content_lengths):.1f} 字符")
    print(f"最小长度: {min(content_lengths)} 字符")
    print(f"最大长度: {max(content_lengths)} 字符")
    
    # 长度分布直方图数据
    length_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
    hist = [0] * (len(length_bins)-1)
    
    for length in content_lengths:
        for i in range(len(length_bins)-1):
            if length_bins[i] <= length < length_bins[i+1]:
                hist[i] += 1
                break
    
    print("\n文本长度分布直方图:")
    for i in range(len(hist)):
        lower = length_bins[i]
        upper = length_bins[i+1] if i < len(length_bins)-2 else "以上"
        print(f"{lower}-{upper}: {hist[i]} 样本 ({hist[i]/len(content_lengths):.2%})")
    
    # 5. 类别平衡性分析
    print("\n=== 类别平衡性分析 ===")
    if len(difficulty_counts) > 1:
        max_count = max(difficulty_counts.values())
        imbalance_ratios = {k: max_count/v for k, v in difficulty_counts.items()}
        print("类别不平衡比例 (最大类别/当前类别):")
        for k, v in sorted(imbalance_ratios.items()):
            print(f"{k}: {v:.2f}x")
    else:
        print("只有一个类别，无法计算平衡性")
    
    # 6. 示例样本展示
    print("\n=== 示例样本展示 ===")
    print("随机展示3个样本及其难度:")
    import random
    for _ in range(3):
        idx = random.randint(0, len(all_contents)-1)
        print(f"\n难度: {all_difficulties[idx]}")
        print(f"内容(前100字符): {all_contents[idx][:100]}...")
        print(f"长度: {len(all_contents[idx])} 字符")