import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# EL TA3DEEL HENA: 7atena Lab_3 abl el imports
from Lab_3.utils import get_logger
from Lab_3.models import SimpleMLP, CustomCNN, get_transfer_model

logger = get_logger("Lab3_Trainer")

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            running_loss += loss.item()
            _, pred = out.max(1)
            total += lbls.size(0)
            correct += pred.eq(lbls).sum().item()
    return running_loss / len(loader), correct / total

def train_loop(model, optimizer, criterion, train_loader, test_loader, epochs, phase_name, device):
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"{phase_name} Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        logger.info(f"[{phase_name}] Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

    return train_losses, test_losses, test_acc

def evaluate_best_model(model, test_loader, classes, device):
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out = model(imgs.to(device))
            all_preds.extend(out.max(1)[1].cpu().numpy())
            all_lbls.extend(lbls.numpy())

    report_dict = classification_report(all_lbls, all_preds, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    logger.info("\n\nFull Evaluation Matrix DataFrame:\n" + df_report.round(4).to_markdown())
    return df_report
