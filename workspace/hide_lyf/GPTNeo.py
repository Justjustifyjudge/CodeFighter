import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM

# 配置模型参数
config = GPTNeoConfig(
    vocab_size=50257,
    n_positions=2048,
    n_embd=4096,
    n_layer=24,
    n_head=16,
    rotary_dim=64,
    activation_function="gelu_new",
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=50256,
    eos_token_id=50256,
    tie_word_embeddings=False
)

# 构建模型
model = GPTNeoForCausalLM(config)

# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {total_params}")

from transformers import GPT2Tokenizer

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 示例数据
text = "这是一个示例句子。"
input_ids = tokenizer.encode(text, return_tensors='pt')

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 示例数据集
data = [input_ids]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
device = [6,7]
model.to(device)
model.train()

for epoch in range(3):
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

model.eval()
total_loss = 0
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        total_loss += loss.item()

perplexity = torch.exp(torch.tensor(total_loss / len(dataloader)))
print(f"Perplexity: {perplexity}")