import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    # take the training and test images from the directories
    train_dir = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train/seg_train'
    test_dir = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test'

    # Normalization to match ImageNet stats for ResNet
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Lazy Loading: PyTorch will read paths without overloading the RAM
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # DataLoaders: retrieve the images during training time 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes
