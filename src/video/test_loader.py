from dataset import get_dataloaders

train_loader, val_loader = get_dataloaders()

print(f"✅ Loaded {len(train_loader.dataset)} training images")
print(f"✅ Loaded {len(val_loader.dataset)} validation images")

# Check one batch
for batch in train_loader:
    images, labels = batch
    print("Images shape:", images.shape)   # Should be [B, 3, 224, 224]
    print("Labels:", labels)
    break
