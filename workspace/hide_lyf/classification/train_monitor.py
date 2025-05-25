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
import wandb
import random

# 设置随机种子保证可重复性
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
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
        
        wandb.log({"batch_train_loss": loss.item()})
        progress_bar.set_postfix({'loss': np.mean(losses[-10:])})
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def is_correct_prediction(true_label, pred_label):
    # 如果预测正确，直接返回True
    if true_label == pred_label:
        return True
    # 如果真实标签是easy且预测为medium，返回True
    if true_label == 0 and pred_label == 1:  # 假设0是easy，1是medium
        return True
    # 如果真实标签是medium且预测为easy，返回True
    if true_label == 1 and pred_label == 0:
        return True
    # 其他情况返回False
    return False

def eval_model(model, data_loader, loss_fn, device, n_examples, mode='val'):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc=f'Evaluating {mode}', leave=False)
    
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
            
            # 修改正确预测的计算方式
            batch_correct = 0
            for true_label, pred_label in zip(labels, preds):
                if is_correct_prediction(true_label.item(), pred_label.item()):
                    batch_correct += 1
            correct_predictions += batch_correct
            
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': np.mean(losses[-10:])})
    
    # 计算分类报告（原始版本，不考虑easy-medium互换）
    class_report_dict = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    class_report_str = classification_report(
        all_labels, all_preds, zero_division=0
    )
    
    # 计算调整后的准确率
    adjusted_accuracy = correct_predictions / n_examples
    
    avg_loss = np.mean(losses)
    
    return adjusted_accuracy, avg_loss, class_report_dict, class_report_str

def main():
    set_seed(42)
    
    JSON_FOLDER = "/workspace/hide_lyf/data/leetcode_data_multi"
    BATCH_SIZE = 16
    MAX_LENGTH = 1024
    EPOCHS = 20
    PATIENCE = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        project="difficulty-classifier",
        config={
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "learning_rate": 2e-5,
            "architecture": "Longformer",
            "dataset": "LeetCode Problems",
            "special_case": "easy-medium interchangeable"
        }
    )
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    dataset = create_data_loader(JSON_FOLDER, tokenizer, BATCH_SIZE, MAX_LENGTH).dataset
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # train_dataset, val_dataset, test_dataset = random_split(
    #     dataset, [train_size, val_size, test_size]
    # )
    train_dataset=dataset
    val_dataset=dataset
    test_dataset=dataset
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DifficultyClassifier(dataset.num_classes).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    
    best_accuracy = 0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE, scheduler, len(train_dataset)
        )
        
        val_acc, val_loss, val_report_dict, val_report_str = eval_model(
            model, val_loader, loss_fn, DEVICE, len(val_dataset), mode='val'
        )
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_report_dict['weighted avg']['precision'],
            "val_recall": val_report_dict['weighted avg']['recall'],
            "val_f1": val_report_dict['weighted avg']['f1-score'],
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f"Val Classification Report (original):\n{val_report_str}")
        
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
    
    model.load_state_dict(best_model_wts)
    
    test_acc, test_loss, test_report_dict, test_report_str = eval_model(
        model, test_loader, loss_fn, DEVICE, len(test_dataset), mode='test'
    )
    
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_report_dict['weighted avg']['precision'],
        "test_recall": test_report_dict['weighted avg']['recall'],
        "test_f1": test_report_dict['weighted avg']['f1-score']
    })
    
    print(f'\nTest Results:')
    print(f'Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    print(f"Test Classification Report (original):\n{test_report_str}")
    
    torch.save(model.state_dict(), 'difficulty_classifier_final.bin')
    # wandb.save('difficulty_classifier_final.bin')
    
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
    
    wandb.finish()

if __name__ == "__main__":
    main()