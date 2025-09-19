# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 00:55:18 2025

@author: Zhao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from itertools import cycle
def trainEE_KL(
    model, 
    id_dataloader, 
    id_testloader,
    ood_dataloader, 
    exit_number,
    criterion_id, 
    optimizer,
    stepsize = 30,
    num_epochs=50,
    alpha = 0.5
):
    if stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,50])
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            model.train()  # Set model to training mode
            running_loss = 0.0
            id_corrects = 0
            ood_certainty = []



            # Mix ID and OOD data in training batches
            for (id_inputs, id_labels), (ood_inputs, _) in zip(id_dataloader,  cycle(ood_dataloader)):
                # Concatenate ID and OOD samples
                inputs = torch.cat([id_inputs, ood_inputs], dim=0).to('cuda')
                labels = torch.cat([id_labels, torch.zeros(len(ood_inputs))], dim=0).to('cuda')  # Dummy labels for OOD
                is_id = torch.cat([torch.ones(len(id_inputs)), torch.zeros(len(ood_inputs))], dim=0).to('cuda')

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass through the specified early exit
                if exit_number == 1:
                    outputs, _ = model(inputs, return_exit1=True)
                elif exit_number == 2:
                    outputs, _ = model(inputs, return_exit2=True)
                else:
                       
                    outputs,_ = model(inputs)

                # Compute the loss
                ce_loss = criterion_id(outputs[:len(id_inputs)], labels[:len(id_inputs)].long())  # Ensurelabels are Long

                probs=F.softmax(outputs[len(id_inputs):],dim=1)
                uniform_dist= torch.ones_like(probs)/probs.size(1)
                kl_div = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[len(id_inputs):], dim=1), uniform_dist)
                alpha = 0.5

                loss = torch.mean(is_id * ce_loss + alpha * (1 - is_id) * kl_div)    

                # Backward + optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, id_preds = torch.max(outputs[:len(id_inputs)], 1)
                id_corrects += torch.sum(id_preds == labels[:len(id_inputs)].long()).item()
                ood_certainty.append(torch.mean(F.softmax(outputs[len(id_inputs):], dim=1).max(dim=1).values).item())

            epoch_loss = running_loss / (len(id_dataloader.dataset) + len(ood_dataloader.dataset))
            epoch_id_acc = id_corrects / len(id_dataloader.dataset)
            avg_ood_certainty = sum(ood_certainty) / len(ood_certainty)
            test(model,criterion_id,id_testloader)

            print(f'Train Loss: {epoch_loss:.4f} | ID Acc: {epoch_id_acc:.4f} | OOD Certainty: {avg_ood_certainty:.4f}')
            if stepsize> 0: scheduler.step()
            
def trainEE(
    model, 
    dataloader, 
    testloader,
    exit_number,
    criterion, 
    optimizer,
    stepsize = 30,
    num_epochs=50,
):
    if stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 50])
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        id_corrects = 0


        for inputs, labels in dataloader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')         
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the specified early exit
            if exit_number == 1:
                outputs, _ = model(inputs, return_exit1=True)
            elif exit_number == 2:
                outputs, _ = model(inputs, return_exit2=True)
            else:
                outputs, _ = model(inputs)

            # Compute the loss using only cross-entropy loss
            ce_loss = criterion(outputs, labels.long())  # Ensure labels are Long

            loss = torch.mean(ce_loss)  # Use only cross-entropy loss

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, id_preds = torch.max(outputs[:len(inputs)], 1)
            id_corrects += torch.sum(id_preds == labels.long()).item()
 
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_id_acc = id_corrects / len(dataloader.dataset)
        test(model, criterion, testloader)

        print(f'Train Loss: {epoch_loss:.4f} | ID Acc: {epoch_id_acc:.4f} ')
        
        if stepsize > 0:
            scheduler.step()
            
def test(net, criterion, testloader, epoch=None):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()


    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                outputs,_ = net(data)
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels.data).sum()


    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('test Acc: {:.5f}'.format(acc))