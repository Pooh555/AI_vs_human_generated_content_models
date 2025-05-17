import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from dataset import get_dataloaders
from model import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
train_loader, val_loader = get_dataloaders(batch_size=32)

# Tracking
best_acc = 0
val_accuracies = []
train_losses = []

# Training Loop
for epoch in range(1, 6):  # 5 epochs
    print(f"\n🔁 Epoch {epoch}")
    model.train()
    train_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu()
            preds += (probs > 0.5).int().tolist()
            truths += labels.tolist()

    acc = accuracy_score(truths, preds)
    avg_loss = train_loss / len(train_loader)

    val_accuracies.append(acc)
    train_losses.append(avg_loss)

    print(f"📊 Validation Accuracy: {acc:.4f} | Loss: {avg_loss:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ New best model saved at epoch {epoch}!")

# 📈 Plot Accuracy and Loss
epochs = list(range(1, len(val_accuracies) + 1))
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, val_accuracies, marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, marker='x', color='red')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()
