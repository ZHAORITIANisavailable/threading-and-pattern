import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def accuracy_EE_byclass(model, dataloader, id_classes=None, ood_classes=None, return_exit1=False, return_exit2=False, device='cuda'):
    import torch
    import numpy as np
    from collections import defaultdict

    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    all_preds = []
    all_labels = []
    all_exits = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if return_exit1:
                outputs, _ = model(images, return_exit1=True)
                exit_id = 1
            elif return_exit2:
                outputs, _ = model(images, return_exit2=True)
                exit_id = 2
            else:
                outputs, exit_ids = model(images)
                exit_id = 3  # main exit or auto-select

            preds = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                all_preds.append(preds[i].item())
                all_labels.append(labels[i].item())
                all_exits.append(exit_id if isinstance(exit_id, int) else exit_ids[i].item())
                correct[labels[i].item()] += int(preds[i].item() == labels[i].item())
                total[labels[i].item()] += 1

    # Print overall accuracy
    overall_acc = np.mean([p == t for p, t in zip(all_preds, all_labels)])
    print(f"\n=== Accuracy for Exit {exit_id} ===")
    print(f"Overall Accuracy: {overall_acc:.2%}")

    # Print per-class accuracy with class names
    print("Per-class accuracy:")
    for cls in sorted(total.keys()):
        acc = correct[cls] / total[cls] if total[cls] > 0 else 0
        name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"  {name:<12}: {acc:.2%} ({correct[cls]}/{total[cls]})")

    # If id_classes and ood_classes are provided, print ID/OOD accuracy
    if id_classes is not None and ood_classes is not None:
        id_indices = [i for i, label in enumerate(all_labels) if label in id_classes]
        ood_indices = [i for i, label in enumerate(all_labels) if label in ood_classes]
        id_acc = np.mean([all_preds[i] == all_labels[i] for i in id_indices]) if id_indices else 0
        ood_acc = np.mean([all_preds[i] == all_labels[i] for i in ood_indices]) if ood_indices else 0
        id_names = [class_names[i] for i in id_classes if i < len(class_names)]
        ood_names = [class_names[i] for i in ood_classes if i < len(class_names)]
        print(f"ID Accuracy: {id_acc:.2%} (classes: {id_names})")
        print(f"OOD Accuracy: {ood_acc:.2%} (classes: {ood_names})")

    # Return a summary dictionary
    return {
        "overall_acc": overall_acc,
        "per_class_acc": {class_names[cls] if cls < len(class_names) else str(cls): correct[cls] / total[cls] if total[cls] > 0 else 0 for cls in total},
        "id_acc": id_acc if id_classes is not None and ood_classes is not None else None,
        "ood_acc": ood_acc if id_classes is not None and ood_classes is not None else None,
        "exit": exit_id
    }

def evaluate_and_log(model, dataloader, thresholds=(0.8872, 0.5428), device='cuda', force_exit=None):
    """Evaluate model and log detailed results."""
    model.eval()
    logs = {
        'logits': [],
        'confs': [],
        'preds': [],
        'labels': [],
        'exits': [],
    }

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Handle forced exit cases
            if force_exit == 1:
                outputs, _ = model(images, return_exit1=True)
                exit_ids = torch.ones(len(images), dtype=torch.long)
            elif force_exit == 2:
                outputs, _ = model(images, return_exit2=True)
                exit_ids = torch.ones(len(images), dtype=torch.long) * 2
            elif force_exit == 3:
                outputs, _ = model(images)
                exit_ids = torch.ones(len(images), dtype=torch.long) * 3
            else:
                # Use automatic early exit mode
                outputs, exit_ids = model(
                    images,
                    auto_select=True,
                    threshold1=thresholds[0],
                    threshold2=thresholds[1]
                )

            # Get softmax probabilities and confidence predictions
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # Log per-sample results
            logs['logits'].extend(outputs.cpu())
            logs['confs'].extend(confs.cpu())
            logs['preds'].extend(preds.cpu())
            logs['labels'].extend(labels.cpu())
            logs['exits'].extend(exit_ids.cpu())

    return logs

def extract_exit_dataset(logs, exit_id):
    """Extract dataset for a specific exit."""
    samples = [
        (log, pred) for log, pred, eid in zip(logs['logits'], logs['preds'], logs['exits']) if eid == exit_id
    ]
    if not samples:
        return None
    logits, preds = zip(*samples)
    logits = torch.stack(logits)
    preds = torch.tensor(preds)
    return TensorDataset(logits, preds)

def summarize_exits(logs):
    """Print summary statistics for each exit."""
    exit_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    total = len(logs['labels'])

    for pred, label, exit_id in zip(logs['preds'], logs['labels'], logs['exits']):
        exit_counts[exit_id.item()] += 1
        if pred == label:
            correct_counts[exit_id.item()] += 1

    print("\nExit Summary:")
    for exit_id in sorted(exit_counts.keys()):
        count = exit_counts[exit_id]
        correct = correct_counts[exit_id]
        acc = correct / count if count > 0 else 0
        print(f"  Exit {exit_id}: {count} samples, Accuracy = {acc:.2%}")

    overall_acc = sum(p == t for p, t in zip(logs['preds'], logs['labels'])) / total
    print(f"   Overall Accuracy: {overall_acc:.2%}")

def plot_exit_distribution(logs):
    """Plot histogram of exit usage."""
    counts = Counter([eid.item() for eid in logs['exits']])
    exits = sorted(counts.keys())
    values = [counts[e] for e in exits]

    plt.figure(figsize=(6, 4))
    plt.bar([f"Exit {e}" for e in exits], values, color='skyblue')
    plt.title("Sample Distribution Across Exits")
    plt.xlabel("Exit Used")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

def per_class_accuracy(logs, class_names):
    """Calculate and print per-class accuracy."""
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for pred, label in zip(logs['preds'], logs['labels']):
        cls = label.item()
        total_per_class[cls] += 1
        if pred == label:
            correct_per_class[cls] += 1

    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        total = total_per_class[i]
        correct = correct_per_class[i]
        acc = correct / total if total > 0 else 0
        print(f"  {name:<12}: {acc:.2%} ({correct}/{total})")

def exit_class_heatmap(logs, class_names):
    """Create heatmap visualization of exit usage by class."""
    exit_class_counts = defaultdict(lambda: defaultdict(int))

    for label, exit_id in zip(logs['labels'], logs['exits']):
        exit_class_counts[exit_id.item()][label.item()] += 1

    exits = sorted(exit_class_counts.keys())
    matrix = []
    for eid in exits:
        row = [exit_class_counts[eid][i] for i in range(len(class_names))]
        matrix.append(row)

    plt.figure(figsize=(10, 4))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=[f"Exit {e}" for e in exits])
    plt.xlabel("Class")
    plt.ylabel("Exit Used")
    plt.title("Heatmap of Class Distribution per Exit")
    plt.tight_layout()
    plt.show()

def print_exit_byclass_table(logs, class_names):
    """Print detailed table of exit usage by class."""
    exit_class_counts = defaultdict(lambda: defaultdict(int))
    for label, exit_id in zip(logs['labels'], logs['exits']):
        exit_class_counts[exit_id.item()][label.item()] += 1

    exits = sorted(exit_class_counts.keys())

    header = f"{'Class':<12} | " + " | ".join([f"Exit {e:<3}" for e in exits])
    print("\nExit-Class Distribution Table")
    print(header)
    print("-" * len(header))

    for i, cls_name in enumerate(class_names):
        row = f"{cls_name:<12} | "
        row += " | ".join([f"{exit_class_counts[e].get(i, 0):<7}" for e in exits])
        print(row)
