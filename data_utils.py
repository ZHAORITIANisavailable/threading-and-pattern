import random
from torch.utils.data import DataLoader, Subset, TensorDataset
from itertools import cycle

def split_cifar10_dataset():
    # TODO: Import this function from spliter.py
    pass

def filter_by_samples(dataset, classes_ratios, seed=None):
    rng = random.Random(seed)
    indices = []
    
    # Get class to index mapping
    if hasattr(dataset, 'class_to_idx'):
        class_to_idx = dataset.class_to_idx
    else:
        class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}
    
    # Collect all sample labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'samples'):
        labels = [label for (_, label) in dataset.samples]
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
    
    # Process each class
    for cls, ratio in classes_ratios.items():
        if cls not in class_to_idx:
            raise ValueError(f"Class {cls} not found in dataset.")
        
        class_idx = class_to_idx[cls]
        class_indices = [i for i, label in enumerate(labels) if label == class_idx]
        
        if not class_indices:
            continue
        
        # Calculate sample count and ensure validity
        num_samples = int(len(class_indices) * ratio)
        num_samples = max(0, min(num_samples, len(class_indices)))
        
        if num_samples == 0:
            continue
        
        # Random sampling
        selected = rng.sample(class_indices, num_samples)
        indices.extend(selected)
    
    # Shuffle overall order
    rng.shuffle(indices)
    return Subset(dataset, indices)

def filter_by_classes(dataset, classes_to_include):
    """Filter dataset to only include specified classes."""
    class_indices = [dataset.class_to_idx[cls] for cls in classes_to_include]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in class_indices]
    return Subset(dataset, indices)

def create_subset_dataloaders(train_dataset, test_dataset, classes_subset, batch_size=64, num_workers=4):
    """Create train and test dataloaders for a subset of classes."""
    subset = {
        'train': filter_by_classes(train_dataset, classes_subset),
        'test': filter_by_classes(test_dataset, classes_subset)
    }
    
    dataloaders = {
        'train': DataLoader(subset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': DataLoader(subset['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloaders
