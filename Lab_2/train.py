import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from utils import set_seed, get_logger
from models import SimpleModel, ComplexModel

set_seed(42)
logger = get_logger("GoldenRules")

# using gpu for fatser traininng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# normalizng as 0-255 is a huge range, and the gradients will explode if the step is not done.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Using 15,000 samples (25% of dataset). Large enough for a real experiment, 
# but small enough to force overfitting in 20 epochs.
subset_indices = torch.randperm(len(full_train_set))[:15000]
train_set = Subset(full_train_set, subset_indices)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

def evaluate(model, loader, criterion):
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

def train_loop(model, optimizer, criterion, epochs, phase_name, l1_lambda=0.0):
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if l1_lambda > 0:
                loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        logger.info(f"[{phase_name}] Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        
    return train_losses, test_losses, test_acc

crit = nn.CrossEntropyLoss()
history_dict = {}

# GOLDEN RULES 

# first rule is the sanity check, using one batch to overfit and chek if we will get 100% accuracy 
logger.info("\n\n--- Phase 1: Sanity Check ---")
sanity_model = ComplexModel().to(device)
opt1 = optim.Adam(sanity_model.parameters(), lr=1e-3)
batch_imgs, batch_lbls = next(iter(train_loader))
batch_imgs, batch_lbls = batch_imgs.to(device), batch_lbls.to(device)
sanity_model.train()
for epoch in range(50):
    opt1.zero_grad()
    out = sanity_model(batch_imgs)
    loss = crit(out, batch_lbls)
    loss.backward()
    opt1.step()
_, preds = out.max(1)
logger.info(f"Sanity Check Accuracy on 1 batch: {preds.eq(batch_lbls).sum().item() / batch_lbls.size(0):.4f}")


# second rule run a simple model with few epochs, most prop it will underfit 
logger.info("\n\n--- Phase 2: Baseline (Underfitting) ---")
simple = SimpleModel().to(device)
opt2 = optim.Adam(simple.parameters(), lr=1e-3)
h_train, h_test, _ = train_loop(simple, opt2, crit, epochs=5, phase_name="Baseline")
history_dict['baseline'] = {'train': h_train, 'test': h_test}


# third rule train a very complex model and see whether the model will overfit or not 
logger.info("\n\n--- Phase 3: Reduce Bias (Overfitting) ---")
complex_model = ComplexModel(use_dropout=False).to(device)
opt3 = optim.Adam(complex_model.parameters(), lr=1e-3)
# 30 epochs to clearly show train accuracy vs test accuracy gap
h_train, h_test, _ = train_loop(complex_model, opt3, crit, epochs=30, phase_name="Overfit")
history_dict['overfit'] = {'train': h_train, 'test': h_test}

# fourth trying diff soluctions combining L1/L2 with dropouts on the SAME complex model
logger.info("\n\n--- Phase 4: Reduce Variance (Testing 5 Strategies) ---")
reg_epochs = 40

# 1. L2 Only
logger.info("Trying L2 Only...")
model_l2 = ComplexModel(use_dropout=False).to(device)
opt_l2 = optim.Adam(model_l2.parameters(), lr=1e-3, weight_decay=1e-4) 
h_train_l2, h_test_l2, acc_l2 = train_loop(model_l2, opt_l2, crit, epochs=reg_epochs, phase_name="L2_Only")
history_dict['l2'] = {'train': h_train_l2, 'test': h_test_l2}

# 2. L1 Only
logger.info("\nTrying L1 Only...")
model_l1 = ComplexModel(use_dropout=False).to(device)
opt_l1 = optim.Adam(model_l1.parameters(), lr=1e-3)
h_train_l1, h_test_l1, acc_l1 = train_loop(model_l1, opt_l1, crit, epochs=reg_epochs, phase_name="L1_Only", l1_lambda=1e-5)
history_dict['l1'] = {'train': h_train_l1, 'test': h_test_l1}

# 3. Dropout (0.5) Only
logger.info("\nTrying Dropout (0.5) Only...")
model_drop5 = ComplexModel(use_dropout=True).to(device)
model_drop5.dropout = nn.Dropout(0.5)
opt_drop5 = optim.Adam(model_drop5.parameters(), lr=1e-3)
h_train_drop5, h_test_drop5, acc_drop5 = train_loop(model_drop5, opt_drop5, crit, epochs=reg_epochs, phase_name="Drop_0.5")
history_dict['drop_5'] = {'train': h_train_drop5, 'test': h_test_drop5}

# 4. L2 + Dropout (0.4)
logger.info("\nTrying L2 + Dropout (0.4)...")
model_l2_drop4 = ComplexModel(use_dropout=True).to(device)
model_l2_drop4.dropout = nn.Dropout(0.4)
opt_l2_drop4 = optim.Adam(model_l2_drop4.parameters(), lr=1e-3, weight_decay=1e-5)
h_train_l2_drop4, h_test_l2_drop4, acc_l2_drop4 = train_loop(model_l2_drop4, opt_l2_drop4, crit, epochs=reg_epochs, phase_name="L2+Drop_0.4")
history_dict['l2_drop4'] = {'train': h_train_l2_drop4, 'test': h_test_l2_drop4}

# 5. L1 + Dropout (0.4)
logger.info("\nTrying L1 + Dropout (0.4)...")
model_l1_drop4 = ComplexModel(use_dropout=True).to(device)
model_l1_drop4.dropout = nn.Dropout(0.4)
opt_l1_drop4 = optim.Adam(model_l1_drop4.parameters(), lr=1e-3)
h_train_l1_drop4, h_test_l1_drop4, acc_l1_drop4 = train_loop(model_l1_drop4, opt_l1_drop4, crit, epochs=reg_epochs, phase_name="L1+Drop_0.4", l1_lambda=1e-5)
history_dict['l1_drop4'] = {'train': h_train_l1_drop4, 'test': h_test_l1_drop4}


# final step is evaluating models and choosing best one 
logger.info("\n\n--- Phase 5: Selection & DataFrame Evaluation ---")
# Comparing all 5 models
best_acc = max(acc_l2, acc_l1, acc_drop5, acc_l2_drop4, acc_l1_drop4)

if best_acc == acc_l2:
    best_model, model_name = model_l2, "L2 Only Model"
elif best_acc == acc_l1:
    best_model, model_name = model_l1, "L1 Only Model"
elif best_acc == acc_drop5:
    best_model, model_name = model_drop5, "Dropout (0.5) Only Model"
elif best_acc == acc_l2_drop4:
    best_model, model_name = model_l2_drop4, "L2 + Dropout (0.4) Model"
else:
    best_model, model_name = model_l1_drop4, "L1 + Dropout (0.4) Model"

logger.info(f"Selected {model_name} with Test Acc: {best_acc:.4f}")

# print evaluation matrix in a shape of a dataframe to be a better looking
best_model.eval()
all_preds, all_lbls = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = best_model(imgs.to(device))
        all_preds.extend(out.max(1)[1].cpu().numpy())
        all_lbls.extend(lbls.numpy())

report_dict = classification_report(all_lbls, all_preds, target_names=test_set.classes, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
logger.info("\n\nFull Evaluation Matrix DataFrame:\n" + df_report.round(4).to_markdown())

torch.save(history_dict, "history.pt")
