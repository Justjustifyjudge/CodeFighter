import torch
from transformers import LongformerTokenizer
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import classification_report
from dataloader import create_data_loader
from model import DifficultyClassifier
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
import copy
from torch.utils.data import Dataset, DataLoader

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    # 添加进度条
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 更新进度条信息
        progress_bar.set_postfix({'loss': np.mean(losses[-10:])})
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    # 添加进度条
    progress_bar = tqdm(data_loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for d in progress_bar:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            # 更新进度条信息
            progress_bar.set_postfix({'loss': np.mean(losses[-10:])})
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    # 参数设置
    JSON_FOLDER = "/workspace/hide_lyf/data/leetcode_data_multi"
    BATCH_SIZE = 16
    MAX_LENGTH = 1024
    EPOCHS = 50
    PATIENCE = 3  # 早停耐心值
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化tokenizer和模型
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    dataset = create_data_loader(JSON_FOLDER, tokenizer, BATCH_SIZE, MAX_LENGTH).dataset
    
    # 拆分训练集和验证集 (90%训练，10%验证)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 获取类别数量
    model = DifficultyClassifier(dataset.num_classes).to(DEVICE)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    
    # 早停相关变量
    best_accuracy = 0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # 训练循环
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        # 训练阶段
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            DEVICE,
            scheduler,
            len(train_dataset)
        )
        
        # 验证阶段
        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            DEVICE,
            len(val_dataset)
        )
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 早停检查
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.bin')
            print(f'New best model saved! (Accuracy: {best_accuracy:.4f})')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter}/{PATIENCE} epochs')
            
            if patience_counter >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch+1} epochs!')
                print(f'Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch+1}')
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'difficulty_classifier_final.bin')
    
    # 测试模型
    test_text = "这是一个测试问题内容..."
    encoded_text = tokenizer.encode_plus(
        test_text,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_text['input_ids'].to(DEVICE)
    attention_mask = encoded_text['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        predicted_label = dataset.label_encoder.inverse_transform(prediction.cpu())
        print(f"\nPredicted difficulty: {predicted_label[0]}")

if __name__ == "__main__":
    main()