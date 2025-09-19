# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:08:23 2025

@author: Zhao
"""
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def evaluate_EE_byclass(model, testloader, id_classes, ood_classes, return_exit1=False, return_exit2=False):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track accuracy per class
    correct_preds = {classname: 0 for classname in class_names}
    total_preds = {classname: 0 for classname in class_names}

    # Initialize variables to track accuracy for ID and OOD classes
    correct_id = 0
    total_id = 0
    correct_ood = 0
    total_ood = 0

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs, _ = model(inputs, return_exit1, return_exit2)
            _, preds = torch.max(outputs, 1)

            # Track accuracy for each class
            for label, pred in zip(labels, preds):
                classname = class_names[label]
                if pred == label:
                    correct_preds[classname] += 1
                    if classname in id_classes:
                        correct_id += 1
                    elif classname in ood_classes:
                        correct_ood += 1
                total_preds[classname] += 1
                if classname in id_classes:
                    total_id += 1
                elif classname in ood_classes:
                    total_ood += 1

    # Calculate and print accuracy for each class
    for classname, correct_count in correct_preds.items():
        accuracy = 100 * float(correct_count) / total_preds[classname]
        print(f'Accuracy for class {classname}: {accuracy:.2f}%')

    # Calculate and print accuracy for ID and OOD classes
    if total_id > 0:
        id_accuracy = 100 * float(correct_id) / total_id
        print(f'Accuracy for ID classes: {id_accuracy:.2f}%')
    else:
        print('No samples from ID classes.')

    if total_ood > 0:
        ood_accuracy = 100 * float(correct_ood) / total_ood
        print(f'Accuracy for OOD classes: {ood_accuracy:.2f}%')
    else:
        print('No samples from OOD classes.')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn.functional as F
import numpy as np

def softmax_ood_detector_with_roc_and_distribution(
    model, 
    id_dataloader, 
    ood_dataloader, 
    exit_number=None,
    device='cuda', 
    invert_scores=True 
):
    """
    Perform OOD detection using softmax-based maximum probability and plot both 
    ROC curve and score distributions.
    """
    model.eval()  # Set model to evaluation mode
    id_scores = []
    ood_scores = []

    # Process ID dataset
    with torch.no_grad():
        for inputs, _ in id_dataloader:
            inputs = inputs.to(device)
            # Forward pass through the specified early exit
            if exit_number == 1:
                outputs, _ = model(inputs, return_exit1=True)
            elif exit_number == 2:
                outputs, _ = model(inputs, return_exit2=True)
            else:
                outputs, _ = model(inputs)
            # Compute softmax probabilities
            softmax_probs = F.softmax(outputs, dim=1)
            max_probs, _ = softmax_probs.max(dim=1)  # Maximum probability for each sample
            # Append ID scores
            id_scores.extend(max_probs.cpu().numpy())

    # Process OOD dataset
    with torch.no_grad():
        for inputs, _ in ood_dataloader:
            inputs = inputs.to(device)
            # Forward pass through the specified early exit
            if exit_number == 1:
                outputs, _ = model(inputs, return_exit1=True)
            elif exit_number == 2:
                outputs, _ = model(inputs, return_exit2=True)
            else:
                outputs, _ = model(inputs)
            # Compute softmax probabilities
            softmax_probs = F.softmax(outputs, dim=1)
            max_probs, _ = softmax_probs.max(dim=1)  # Maximum probability for each sample
            # Append OOD scores
            ood_scores.extend(max_probs.cpu().numpy())

    # Combine scores for ROC curve
    scores = id_scores + ood_scores
    labels = [0] * len(id_scores) + [1] * len(ood_scores)  # 0 for ID, 1 for OOD
    
    if invert_scores:
        scores = np.array(scores)
        scores = -scores
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Find FPR at 95% TPR
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))  # Find the closest TPR to 95%
    fpr_at_95_tpr = fpr[idx]
    threshold_at_95_tpr = -thresholds[idx]

    print(f"FPR at 95% TPR: {fpr_at_95_tpr:.4f}")
    print(f"Threshold at 95% TPR: {threshold_at_95_tpr:.4f}")

    # Compute optimal threshold using Youden's index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)  # Index of the optimal threshold
    optimal_threshold = -thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr_at_95_tpr, target_tpr, color='red', label=f'FPR@95%TPR = {fpr_at_95_tpr:.4f}', zorder=5)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Plot score distributions
    plt.figure(figsize=(8, 6))
    plt.hist(id_scores, bins=30, alpha=0.6, color='blue', label='ID Scores', density=True)
    plt.hist(ood_scores, bins=30, alpha=0.6, color='red', label='OOD Scores', density=True)
    plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.xlabel('Softmax Maximum Probability')
    plt.ylabel('Density')
    plt.title('Distribution of ID and OOD Scores')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    
    print(len(id_scores))
    return roc_auc, fpr, tpr