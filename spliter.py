
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ]),
}

class CustomCIFAR10(Dataset):
    def __init__(self, data, targets, transform=None, class_to_idx=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = [(data[i], targets[i]) for i in range(len(data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        if self.transform:
            img = self.transform(img)
        return img, target

def split_cifar10_dataset():
    # Load the CIFAR-10 dataset
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Get the data and targets
    train_data = cifar10_train_dataset.data
    train_targets = cifar10_train_dataset.targets
    test_data = cifar10_test_dataset.data
    test_targets = cifar10_test_dataset.targets
    # Get the class-to-index mapping
    class_to_idx = cifar10_train_dataset.class_to_idx
    # Get the indices for each class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(train_targets):
        class_indices[label].append(idx)

    # Split the indices for each class into two halves
    split_indices_1 = []
    split_indices_2 = []
    for indices in class_indices.values():
        half = len(indices) // 2
        split_indices_1.extend(indices[:half])
        split_indices_2.extend(indices[half:])

    # Create the two training datasets
    train_data_1 = train_data[split_indices_1]
    train_targets_1 = [train_targets[i] for i in split_indices_1]
    train_data_2 = train_data[split_indices_2]
    train_targets_2 = [train_targets[i] for i in split_indices_2]

    train_dataset_1 = CustomCIFAR10(train_data_1, train_targets_1, transform=data_transforms['train'], class_to_idx=class_to_idx)
    train_dataset_2 = CustomCIFAR10(train_data_2, train_targets_2, transform=data_transforms['train'], class_to_idx=class_to_idx)

    # Create the test dataset
    test_dataset = CustomCIFAR10(test_data, test_targets, transform=data_transforms['test'], class_to_idx=class_to_idx)

    return train_dataset_1, train_dataset_2, test_dataset


import random
from torch.utils.data import Subset

def filter_by_ratio(dataset, classes_ratios, seed=None):
    rng = random.Random(seed)
    indices = []
    
    # 确定类别到索引的映射
    if hasattr(dataset, 'class_to_idx'):
        class_to_idx = dataset.class_to_idx
    else:
        class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}
    
    # 收集所有样本的标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'samples'):
        labels = [label for (_, label) in dataset.samples]
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
    
    # 处理每个类别
    for cls, ratio in classes_ratios.items():
        if cls not in class_to_idx:
            raise ValueError(f"Class {cls} not found in dataset.")
        
        class_idx = class_to_idx[cls]
        class_indices = [i for i, label in enumerate(labels) if label == class_idx]
        
        if not class_indices:
            continue
        
        # 计算样本数量并确保有效性
        num_samples = int(len(class_indices) * ratio)
        num_samples = max(0, min(num_samples, len(class_indices)))
        
        if num_samples == 0:
            continue
        
        # 随机抽样
        selected = rng.sample(class_indices, num_samples)
        indices.extend(selected)
    
    # 打乱整体顺序
    rng.shuffle(indices)
    return Subset(dataset, indices)