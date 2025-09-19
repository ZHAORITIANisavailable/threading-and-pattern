import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from itertools import cycle
import threading
from queue import Queue

def trainEE_KL(
    model, 
    id_dataloader, 
    ood_dataloader, 
    exit_number,
    criterion_id, 
    optimizer,
    stepsize = 30,
    num_epochs=50,
    alpha = 0.5,
    id_testloader=None,
    # 新增参数（无需手动传递）
    stop_event=None,     # type: threading.Event
    request_queue=None   # type: Queue
):
    if stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,50])
        
    for epoch in range(num_epochs):
        # 检查停止信号
        if stop_event and stop_event.is_set():
            print("\n[training has been terminated by user]")
            return
            
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        id_corrects = 0
        ood_certainty = []

        # 混合数据批次
        data_iter = zip(id_dataloader, cycle(ood_dataloader))
        for batch_idx, ((id_inputs, id_labels), (ood_inputs, _)) in enumerate(data_iter):
            # 处理队列中的请求（每个batch处理一次）
            if request_queue and not request_queue.empty():
                while not request_queue.empty():
                    sample = request_queue.get()
                    print(f"\n[realtime request] samples siza: {sample.shape}")
                    with torch.no_grad():
                        if exit_number == 1:
                            pred, _ = model(sample.to('cuda'), return_exit1=True)
                        elif exit_number == 2:
                            pred, _ = model(sample.to('cuda'), return_exit2=True)
                        print(f"prediction: {torch.argmax(pred).item()}")

            # 原始训练逻辑
            inputs = torch.cat([id_inputs, ood_inputs], dim=0).to('cuda')
            labels = torch.cat([id_labels, torch.zeros(len(ood_inputs))], dim=0).to('cuda')
            is_id = torch.cat([torch.ones(len(id_inputs)), torch.zeros(len(ood_inputs))], dim=0).to('cuda')

            optimizer.zero_grad()

            if exit_number == 1:
                outputs, _ = model(inputs, return_exit1=True)
            elif exit_number == 2:
                outputs, _ = model(inputs, return_exit2=True)
            else:
                outputs,_ = model(inputs)

            ce_loss = criterion_id(outputs[:len(id_inputs)], labels[:len(id_inputs)].long())
            probs = F.softmax(outputs[len(id_inputs):],dim=1)
            uniform_dist = torch.ones_like(probs)/probs.size(1)
            kl_div = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[len(id_inputs):], dim=1), uniform_dist)
            loss = torch.mean(is_id * ce_loss + alpha * (1 - is_id) * kl_div)    

            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item() * inputs.size(0)
            _, id_preds = torch.max(outputs[:len(id_inputs)], 1)
            id_corrects += torch.sum(id_preds == labels[:len(id_inputs)].long()).item()
            ood_certainty.append(torch.mean(F.softmax(outputs[len(id_inputs):], dim=1).max(dim=1).values).item())

        epoch_loss = running_loss / (len(id_dataloader.dataset) + len(ood_dataloader.dataset))
        epoch_id_acc = id_corrects / len(id_dataloader.dataset)
        avg_ood_certainty = sum(ood_certainty) / len(ood_certainty)
        # 每个epoch结束后进行测试
        if id_testloader is not None:   
            test(model,criterion_id,id_testloader)
        print(f'Train Loss: {epoch_loss:.4f} | ID Acc: {epoch_id_acc:.4f} | OOD Certainty: {avg_ood_certainty:.4f}')
        if stepsize> 0: 
            scheduler.step()

def trainEE_KL_for_profiling(
    model, 
    id_dataloader, 
    ood_dataloader, 
    exit_number,
    criterion_id, 
    optimizer,
    stepsize = 30,
    num_epochs=10, # This will be overridden by profile_steps
    alpha = 0.5,
    # --- PROFILER-SPECIFIC PARAMETERS ---
    prof=None,           # ADDED: To pass the profiler object
    profile_steps=15     # ADDED: To run for a fixed number of steps and then stop
):
    """
    Modified version of trainEE_KL to be used with torch.profiler.
    """
    if stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,50])
        
    # The outer epoch loop is kept, but we'll exit from the inner loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        id_corrects = 0

        data_iter = zip(id_dataloader, cycle(ood_dataloader))
        for batch_idx, ((id_inputs, id_labels), (ood_inputs, _)) in enumerate(data_iter):
            # Your original training logic remains unchanged
            inputs = torch.cat([id_inputs, ood_inputs], dim=0).to('cuda')
            id_labels_on_device = id_labels.to('cuda').long() # Ensure labels are on correct device and type

            optimizer.zero_grad()

            if exit_number == 1:
                outputs, _ = model(inputs, return_exit1=True)
            elif exit_number == 2:
                outputs, _ = model(inputs, return_exit2=True)
            else:
                outputs,_ = model(inputs)

            id_outputs = outputs[:len(id_inputs)]
            ood_outputs = outputs[len(id_inputs):]
            
            ce_loss = criterion_id(id_outputs, id_labels_on_device)
            
            kl_div = F.kl_div(
                F.log_softmax(ood_outputs, dim=1),
                torch.full_like(ood_outputs, 1. / ood_outputs.shape[1]),
                reduction='batchmean'
            )

            loss = ce_loss + alpha * kl_div

            loss.backward()
            optimizer.step()

            # --- PROFILER INTEGRATION ---
            # ADDED: Signal the profiler that a step is complete.
            if prof:
                prof.step()
            # --------------------------
            
            # (Optional) Your statistics calculation code can remain
            # running_loss += loss.item() * inputs.size(0)
            # _, id_preds = torch.max(id_outputs, 1)
            # id_corrects += torch.sum(id_preds == id_labels_on_device).item()
            
            # ADDED: Exit after a few steps to keep the profiling trace small and focused.
            if prof and batch_idx >= profile_steps:
                print(f"\n[Profiler] Reached {profile_steps} steps. Stopping training.")
                return # Exit the function entirely

        # This part will likely not be reached when profiling
        epoch_loss = running_loss / (len(id_dataloader.dataset) + len(ood_dataloader.dataset))
        epoch_id_acc = id_corrects / len(id_dataloader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} | ID Acc: {epoch_id_acc:.4f}')
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

def inference_loop(model, test_loader, model_lock):
    """Run inference loop with thread safety."""
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            with model_lock:
                outputs, exits = model(inputs, auto_select=True)
            print(f"[Inference] Sample {i} completed, exit location: Exit{exits}")

def retrain_loop(model, model_lock, optimizer=None, criterion=None):
    """Run retraining loop with thread safety."""
    if optimizer is None:
        optimizer = optim.SGD(model.early_exit1.parameters(), lr=0.01)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    while True:
        model.train()
        inputs = torch.randn(1, 3, 32, 32).cuda()
        targets = torch.randint(0, 10, (1,)).cuda()
        with model_lock:
            output, _ = model(inputs, return_exit1=True)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("[Training] Trained exit1 once")
