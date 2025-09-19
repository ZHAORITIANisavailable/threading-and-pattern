import threading
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from queue import Queue
import seaborn as sns
from IPython.display import display
import torch.nn.functional as F

from training import trainEE_KL, trainEE_KL_for_profiling



# === 多线程训练控制器封装 ===
class TrainingController:
    def __init__(self):
        self.stop_event = threading.Event()
        self.request_queue = Queue()
        self.train_thread = None

    def start_training(self, train_function, **kwargs):
        """启动训练线程"""
        def _wrapper():
            train_function(
                stop_event=self.stop_event,
                request_queue=self.request_queue,
                **kwargs
            )
        self.stop_event.clear()
        self.train_thread = threading.Thread(target=_wrapper)
        self.train_thread.start()

    def add_request(self, data):
        """添加实时推理请求（暂不用于 A 模式）"""
        self.request_queue.put(data)

    def stop_training(self):
        """停止训练线程"""
        self.stop_event.set()
        if self.train_thread:
            self.train_thread.join()

class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            print(f"{self.name} 未开始计时！")
            return
        self.duration = time.time() - self.start_time
        print(f"[{self.name}] 耗时: {self.duration:.2f} 秒\n")
        return self.duration
    def elapsed(self):
        return round(self.duration, 2)
    
def _visualize_exit_distribution(log_dict, title_suffix=""):
    if not log_dict or all(len(v) == 0 for v in log_dict.values()):
        print(f"[跳过] {title_suffix} 推理数据为空，无法可视化。")
        return

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    exit_class_dist = pd.DataFrame.from_dict(
        {k: Counter(v) for k, v in log_dict.items()},
        orient='index'
    ).fillna(0).astype(int)

    for i in range(len(class_names)):
        if i not in exit_class_dist.columns:
            exit_class_dist[i] = 0
    exit_class_dist = exit_class_dist[sorted(exit_class_dist.columns)]
    exit_class_dist.columns = class_names

    exit_class_dist.T.plot(kind='bar', figsize=(10, 5))
    plt.title(f"Exit-wise Class Distribution {title_suffix}")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.heatmap(exit_class_dist, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Exit-Class Heatmap {title_suffix}")
    plt.ylabel("Exit")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()

    
def analyze_exit_logs_with_accuracy(all_logs, class_names):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(all_logs)

    # ===== 1. 每个出口的总体准确率 =====
    acc_stats = df.groupby('exit').apply(
        lambda g: pd.Series({
            'correct': (g['true'] == g['pred']).sum(),
            'total': len(g),
            'accuracy': (g['true'] == g['pred']).mean()
        })
    )

    print("\n📊 每个出口的总体准确率：")
    print(acc_stats[['correct', 'total', 'accuracy']])

    # ===== 2. 每个出口、每个类别的准确率 =====
    print("\n📚 每个出口的每类准确率（按 true 标签统计）：")
    grouped = df.groupby(['exit', 'true'])

    per_class_acc = {}
    for (exit_num, cls_id), group in grouped:
        correct = (group['pred'] == group['true']).sum()
        total = len(group)
        acc = correct / total if total > 0 else 0.0
        per_class_acc.setdefault(exit_num, {})[class_names[cls_id]] = acc

    # 打印每个出口的 per-class 准确率
    for exit_num in sorted(per_class_acc.keys()):
        print(f"\n📌 Exit {exit_num}：")
        for cls in class_names:
            acc = per_class_acc[exit_num].get(cls, 0.0)
            print(f"  {cls:<12}: {acc:.2%}")

    # ===== 3. 每个出口的预测类别分布（用于可视化） =====
    exit_class_pred = pd.crosstab(df['exit'], df['pred'], rownames=['Exit'], colnames=['Predicted Class'])
    exit_class_pred.columns = [class_names[c] for c in exit_class_pred.columns]

    print("\n📈 每个出口的预测类别分布：")
    display(exit_class_pred)

    # ===== 4. 可视化：柱状图 =====
    exit_class_pred.T.plot(kind='bar', figsize=(10, 5))
    plt.title("Exit-wise Predicted Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.show()

    # ===== 5. 可视化：热力图 =====
    plt.figure(figsize=(8, 4))
    sns.heatmap(exit_class_pred, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Exit-Class Prediction Heatmap")
    plt.ylabel("Exit")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()


def get_system_status(label, process):
    mem = process.memory_info().rss / 1024 ** 2
    threads = process.num_threads()
    return {
        f"Memory {label} (MB)": round(mem, 2),
        f"Threads {label}": threads
    }



def run_patternA_execution(
    model,
    controller1, controller2,
    inference_loader,
    dataloaders_subset1a, dataloaders_subset1b,
    dataloaders_subset2a, dataloaders_subset2b,
    device='cuda',
    visualize=True
):
    results = {}
    process = psutil.Process()

    # === 加载模型后，记录初始状态 ===
    model.load_state_dict(torch.load('model_with_exits_new.pth'))
    results.update(get_system_status("After Model Load", process))
    
    # === 冻结主干，解冻 early exits ===
    for param in model.parameters():
        param.requires_grad = False
    for param in model.early_exit1.parameters():
        param.requires_grad = True
    for param in model.early_exit2.parameters():
        param.requires_grad = True

    optimizer1 = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)

    train_args1 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset1a["train"],
        #'id_testloader': dataloaders_subset1a["test"],
        'ood_dataloader': dataloaders_subset1b["train"],
        'exit_number': 1,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer1,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    train_args2 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset2a["train"],
        #'id_testloader': dataloaders_subset2a["test"],
        'ood_dataloader': dataloaders_subset2b["train"],
        'exit_number': 2,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer2,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    # === 启动训练线程前记录状态 ===
    results.update(get_system_status("Before Training", process))

    print("[PatternA] 启动训练线程 Exit1 和 Exit2...")
    train1_timer = Timer("训练 Exit1 耗时")
    train2_timer = Timer("训练 Exit2 耗时")
    train1_timer.start()
    train2_timer.start()
    controller1.start_training(train_function=train_args1.pop('train_function'), **train_args1)
    controller2.start_training(train_function=train_args2.pop('train_function'), **train_args2)

    # === 推理前记录状态 ===
    results.update(get_system_status("Before Inference", process))

    # === 推理记录结构 ===
    exit_logs = defaultdict(list)
    latency_logs = []
    all_logs = {'true': [], 'pred': [], 'exit': []}

    status_recorded_mid_infer = False

    print("[PatternA] 开始实时推理记录（auto_select=True）...")
    start_infer_time = time.time()

    for batch in inference_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        try:
            start = time.time()
            with torch.no_grad():
                outputs, exit_idx = model(images, auto_select=True, disabled_exits=[1, 2])
            end = time.time()
            latency_logs.append(end - start)

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(exit_idx)):
                exit_number = exit_idx[i].item()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                exit_logs[exit_number].append(true_label)
                all_logs['true'].append(true_label)
                all_logs['pred'].append(pred_label)
                all_logs['exit'].append(exit_number)

            # === 仅记录一次训练过程中的系统状态 ===
            if not status_recorded_mid_infer:
                results.update(get_system_status("during_infer", process))
                status_recorded_mid_infer = True
                time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] 推理失败: {e}")

    infer_duration = time.time() - start_infer_time
    print("\n[推理完成]")

    # === 等待训练线程结束 ===
    controller1.train_thread.join()
    controller2.train_thread.join()
    train1_timer.stop()
    train2_timer.stop()

    # === 最后状态记录 ===
    results.update(get_system_status("After All", process))

    results["Mode"] = "PatternA Execution"
    results["Inference Duration (s)"] = round(infer_duration, 2)
    results["Exit1 Training Time (s)"] = train1_timer.elapsed()
    results["Exit2 Training Time (s)"] = train2_timer.elapsed()
    results["Avg Inference Latency (s)"] = round(np.mean(latency_logs), 4)
    results["Latency Std Dev (s)"] = round(np.std(latency_logs), 4)

    # === 可视化与准确率分析 ===
    if visualize:
        print("\n[可视化] 各出口类别分布")
        _visualize_exit_distribution(exit_logs, title_suffix="(PatternA)")
        analyze_exit_logs_with_accuracy(all_logs, [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ])

    return results,latency_logs, exit_logs


def run_patternB_execution(
    model,
    dataloaders_subset1a,
    dataloaders_subset1b,
    dataloaders_subset2a,
    dataloaders_subset2b,
    controller1,
    controller2,
    inference_loader,
    trainEE_KL,
    visualize=True,
    device='cuda',
):
    process = psutil.Process()
    # Use unified top-level get_system_status

    results = {}
    exit_logs = defaultdict(list)
    all_logs = {'true': [], 'pred': [], 'exit': []}
    latency_logs = []
    inference_logs_phase1 = defaultdict(list)
    inference_logs_phase2 = defaultdict(list)
    for eid in [0, 1, 2]:
        inference_logs_phase1[eid]
        inference_logs_phase2[eid]

    results.update(get_system_status("After Model Load", process))

    current_exit_mode = {'mode': 'exit1_train'}
    stop_flag = {'stop': False}
    switch_requested = {'flag': False}

    def inference_thread():
        print("[PatternB] 启动推理线程...")
        start_infer_time = time.time()
        recorded_midpoint = False
        for batch in inference_loader:
            if switch_requested['flag']:
                print("[PatternB] 检测到模式切换请求，完成当前 batch 后切换...")
                current_exit_mode['mode'] = 'exit2_train'
                switch_requested['flag'] = False
            images, labels = batch[0].to(device), batch[1].to(device)
            try:
                if current_exit_mode['mode'] == 'exit2_train':
                    disabled_exits = [1]
                elif current_exit_mode['mode'] == 'exit1_train':
                    disabled_exits = [2]
                else:
                    disabled_exits = []
                start = time.time()
                with torch.no_grad():
                    outputs, exit_idx = model(images, auto_select=True, disabled_exits=disabled_exits)
                end = time.time()
                latency_logs.append(end - start)
                if not recorded_midpoint:
                    results.update(get_system_status("during_infer", process))
                    recorded_midpoint = True
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                for i in range(len(exit_idx)):
                    exit_number = exit_idx[i].item()
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    exit_logs[exit_number].append(true_label)
                    all_logs['true'].append(true_label)
                    all_logs['pred'].append(pred_label)
                    all_logs['exit'].append(exit_number)
                    if current_exit_mode['mode'] == 'exit1_train':
                        inference_logs_phase1[exit_number].append(true_label)
                    else:
                        inference_logs_phase2[exit_number].append(true_label)
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] 推理失败: {e}")
            if stop_flag['stop']:
                break
        results["Inference Duration (s)"] = round(time.time() - start_infer_time, 2)
        print("[PatternB] 推理线程结束")

    results.update(get_system_status("Before Training", process))
    infer_thread = threading.Thread(target=inference_thread)
    infer_thread.start()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.early_exit1.parameters():
        param.requires_grad = True
    optimizer1 = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    train_args1 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset1a["train"],
        #'id_testloader': dataloaders_subset1a["test"],
        'ood_dataloader': dataloaders_subset1b["train"],
        'exit_number': 1,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer1,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }
    print("[PatternB] 第一阶段：训练 Exit1 (Exit2 和主干处理推理)...")
    train1_timer = Timer("训练 Exit1 耗时")
    train1_timer.start()
    controller1.start_training(train_function=train_args1.pop('train_function'), **train_args1)
    controller1.train_thread.join()
    train1_timer.stop()

    switch_requested['flag'] = True
    for param in model.early_exit1.parameters():
        param.requires_grad = False
    for param in model.early_exit2.parameters():
        param.requires_grad = True
    optimizer2 = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)
    train_args2 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset2a["train"],
        #'id_testloader': dataloaders_subset2a["test"],
        'ood_dataloader': dataloaders_subset2b["train"],
        'exit_number': 2,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer2,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }
    print("[PatternB] 第二阶段：训练 Exit2 (Exit1 和主干处理推理)...")
    train2_timer = Timer("训练 Exit2 耗时")
    train2_timer.start()
    controller2.start_training(train_function=train_args2.pop('train_function'), **train_args2)
    controller2.train_thread.join()
    train2_timer.stop()

    stop_flag['stop'] = True
    infer_thread.join()

    results.update(get_system_status("After All", process))
    results["Mode"] = "PatternB"
    results["Exit1 Training Time (s)"] = train1_timer.elapsed()
    results["Exit2 Training Time (s)"] = train2_timer.elapsed()
    results["Avg Inference Latency (s)"] = round(np.mean(latency_logs), 4)
    results["Latency Std Dev (s)"] = round(np.std(latency_logs), 4)

    if visualize:
        print("\n[可视化] 第一阶段各出口类别分布")
        _visualize_exit_distribution(inference_logs_phase1, title_suffix="(Phase 1)")
        print("\n[可视化] 第二阶段各出口类别分布")
        _visualize_exit_distribution(inference_logs_phase2, title_suffix="(Phase 2)")

        print("\n[可视化] 合并的出口类别分布")
        _visualize_exit_distribution(exit_logs, title_suffix="(Combined)")
        analyze_exit_logs_with_accuracy(all_logs, [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ])
    return results, latency_logs, exit_logs


def run_shared_parallel_execution(
    model,
    controller1, controller2,
    inference_loader,
    dataloaders_subset1a, dataloaders_subset1b,
    dataloaders_subset2a, dataloaders_subset2b,
    trainEE_KL,
    class_names,
    device='cuda',
    visualize=True  # 新增参数
):


    recorded_during_infer = False
    results = {}
    process = psutil.Process()


    results.update(get_system_status("After Model Load", process))

    for param in model.parameters():
        param.requires_grad = False
    for param in model.early_exit1.parameters():
        param.requires_grad = True
    for param in model.early_exit2.parameters():
        param.requires_grad = True

    results.update(get_system_status("Before Training", process))

    optimizer1 = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)

    train_args1 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset1a["train"],
        #'id_testloader': dataloaders_subset1a["test"],
        'ood_dataloader': dataloaders_subset1b["train"],
        'exit_number': 1,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer1,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    train_args2 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset2a["train"],
        #'id_testloader': dataloaders_subset2a["test"],
        'ood_dataloader': dataloaders_subset2b["train"],
        'exit_number': 2,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer2,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    print("[C-Shared] 启动训练线程 Exit1 和 Exit2...")
    train1_timer = Timer("训练 Exit1 耗时")
    train2_timer = Timer("训练 Exit2 耗时")
    train1_timer.start()
    train2_timer.start()
    controller1.start_training(train_function=train_args1.pop('train_function'), **train_args1)
    controller2.start_training(train_function=train_args2.pop('train_function'), **train_args2)

    # === 推理记录结构 ===
    exit_logs = defaultdict(list)
    latency_logs = []
    all_logs = {'true': [], 'pred': [], 'exit': []}

    results.update(get_system_status("Before Inference", process))

    print("[C-Shared] 开始共享模型实时推理记录（auto_select=True）...")

    start_infer_time = time.time()

    for batch in inference_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        try:
            start = time.time()
            with torch.no_grad():
                outputs, exit_idx = model(images, auto_select=True)
            end = time.time()
            latency_logs.append(end - start)

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(exit_idx)):
                exit_number = exit_idx[i].item()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                exit_logs[exit_number].append(true_label)
                all_logs['true'].append(true_label)
                all_logs['pred'].append(pred_label)
                all_logs['exit'].append(exit_number)
            if not recorded_during_infer:
                results.update(get_system_status("during_infer", process))
                recorded_during_infer = True

            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] 推理失败: {e}")

    infer_duration = time.time() - start_infer_time
    print("\n[推理完成]")

    controller1.train_thread.join()
    train1_timer.stop()
    controller2.train_thread.join()
    train2_timer.stop()

    results.update(get_system_status("After All", process))

    # === 系统资源与性能记录 ===
    results["Mode"] = "Shared Parallel Execution"
    results["Inference Duration (s)"] = round(infer_duration, 2)
    results["Exit1 Training Time (s)"] = train1_timer.elapsed()
    results["Exit2 Training Time (s)"] = train2_timer.elapsed()
    results["Avg Inference Latency (s)"] = round(np.mean(latency_logs), 4)
    results["Latency Std Dev (s)"] = round(np.std(latency_logs), 4)

    # === 可视化与准确率分析 ===
    if visualize:
        print("\n[可视化] 各出口类别分布")
        _visualize_exit_distribution(exit_logs, title_suffix="(Shared Mode)")
        analyze_exit_logs_with_accuracy(all_logs, class_names)

    return results, latency_logs, exit_logs

def run_full_suspension_parallel_realtime_logging(
    model,
    controller1, controller2,
    inference_loader,
    dataloaders_subset1a, dataloaders_subset1b,
    dataloaders_subset2a, dataloaders_subset2b,
    device='cuda',
    visualize=True  # 新增参数
):
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

    results = {}
    process = psutil.Process()

    # === 加载模型后，记录初始状态 ===
    model.load_state_dict(torch.load('model_with_exits.pth'))
    results.update(get_system_status("After Model Load", process))
    
    # === 冻结主干，解冻 early exits ===
    for param in model.parameters():
        param.requires_grad = False
    for param in model.early_exit1.parameters():
        param.requires_grad = True
    for param in model.early_exit2.parameters():
        param.requires_grad = True

    optimizer1 = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)

    train_args1 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset1a["train"],
        'id_testloader': dataloaders_subset1a["test"],
        'ood_dataloader': dataloaders_subset1b["train"],
        'exit_number': 1,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer1,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    train_args2 = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset2a["train"],
        'id_testloader': dataloaders_subset2a["test"],
        'ood_dataloader': dataloaders_subset2b["train"],
        'exit_number': 2,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer2,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    # === 启动训练线程前记录状态 ===
    results.update(get_system_status("Before Training", process))

    print("[A-Parallel] 启动训练线程 Exit1 和 Exit2...")
    train1_timer = Timer("训练 Exit1 耗时")
    train2_timer = Timer("训练 Exit2 耗时")
    train1_timer.start()
    train2_timer.start()
    controller1.start_training(train_function=train_args1.pop('train_function'), **train_args1)
    controller2.start_training(train_function=train_args2.pop('train_function'), **train_args2)

    # === 推理前记录状态 ===
    results.update(get_system_status("Before Inference", process))

    # === 推理记录结构 ===
    exit_logs = defaultdict(list)
    latency_logs = []
    all_logs = {'true': [], 'pred': [], 'exit': []}

    status_recorded_mid_infer = False

    # === 启动 profiler 一次性包裹推理阶段 ===
    with profile(
        activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler("./log/trace_full_suspension_infer"),
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
    ) as prof:

        print("[A-Parallel] 开始实时推理记录（auto_select=True）...")
        start_infer_time = time.time()

        for batch in inference_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            try:
                start = time.time()
                with torch.no_grad():
                    outputs, exit_idx = model(images, auto_select=True, disabled_exits=[1, 2])
                end = time.time()
                latency_logs.append(end - start)

                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

                for i in range(len(exit_idx)):
                    exit_number = exit_idx[i].item()
                    true_label = labels[i].item()
                    pred_label = preds[i].item()

                    exit_logs[exit_number].append(true_label)
                    all_logs['true'].append(true_label)
                    all_logs['pred'].append(pred_label)
                    all_logs['exit'].append(exit_number)

                # profiler 步进
                prof.step() 

                # === 仅记录一次训练过程中的系统状态 ===
                if not status_recorded_mid_infer:
                    results.update(get_system_status("during_infer", process))
                    status_recorded_mid_infer = True
                    time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] 推理失败: {e}")

    infer_duration = time.time() - start_infer_time
    print("\n[推理完成]")

    # === 等待训练线程结束 ===
    controller1.train_thread.join()
    controller2.train_thread.join()
    train1_timer.stop()
    train2_timer.stop()

    # === 最后状态记录 ===
    results.update(get_system_status("After All", process))

    results["Mode"] = "Full Suspension (Realtime Logging)"
    results["Inference Duration (s)"] = round(infer_duration, 2)
    results["Exit1 Training Time (s)"] = train1_timer.elapsed()
    results["Exit2 Training Time (s)"] = train2_timer.elapsed()
    results["Avg Inference Latency (s)"] = round(np.mean(latency_logs), 4)
    results["Latency Std Dev (s)"] = round(np.std(latency_logs), 4)

    # === 可视化与准确率分析 ===
    if visualize:
        print("\n[可视化] 各出口类别分布")
        _visualize_exit_distribution(exit_logs, title_suffix="(Phase 1)")
        analyze_exit_logs_with_accuracy(all_logs, [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ])

    return results, latency_logs, exit_logs


def run_pattern_O_no_training(
    model,
    inference_loader,
    class_names,
    device='cuda',
    visualize=True
):
    import time, psutil
    from collections import defaultdict
    import numpy as np
    import torch.nn.functional as F

    results = {}
    process = psutil.Process()

    # Use unified top-level get_system_status

    # === 加载模型 ===
    model.eval()
    results.update(get_system_status("After model load", process))
    # === 推理记录结构 ===
    exit_logs = defaultdict(list)
    latency_logs = []
    all_logs = {'true': [], 'pred': [], 'exit': []}

    print("[Pattern-O] 开始推理记录（无训练线程，auto_select=True）...")

    start_infer_time = time.time()
    recorded_during_infer = False  # 标记是否已记录中途资源状态

    for batch in inference_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        try:
            start = time.time()
            with torch.no_grad():
                outputs, exit_idx = model(images, auto_select=True)
            end = time.time()
            latency_logs.append(end - start)

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(exit_idx)):
                exit_number = exit_idx[i].item()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                exit_logs[exit_number].append(true_label)
                all_logs['true'].append(true_label)
                all_logs['pred'].append(pred_label)
                all_logs['exit'].append(exit_number)
            if not recorded_during_infer:
                results.update(get_system_status("during_infer", process))
                recorded_during_infer = True
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] 推理失败: {e}")

    infer_duration = time.time() - start_infer_time
    print("\n[推理完成]")

    # === 系统资源与性能记录 ===
    results.update(get_system_status("After all", process))

    results["Mode"] = "Pattern-O (No Training)"
    results["Inference Duration (s)"] = round(infer_duration, 2)
    results["Avg Inference Latency (s)"] = round(np.mean(latency_logs), 4)
    results["Latency Std Dev (s)"] = round(np.std(latency_logs), 4)

    if visualize:
        print("\n[可视化] 各出口类别分布：")
        _visualize_exit_distribution(exit_logs, title_suffix="(No Training Mode)")
        analyze_exit_logs_with_accuracy(all_logs, class_names)

    return results, latency_logs, exit_logs



def run_single_exit_training(
    model,
    dataloaders_subset_a,
    dataloaders_subset_b,
    exit_number=1
    
):
    """
    只训练一个 early exit（exit_number=1 或 2），无推理部分。
    """
    results = {}
    process = psutil.Process()

    # === 加载模型后，记录初始状态 ===
    results.update(get_system_status("After Model Load", process))

    # === 冻结主干，解冻指定 early exit ===
    for param in model.parameters():
        param.requires_grad = False
    if exit_number == 1:
        for param in model.early_exit1.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    elif exit_number == 2:
        for param in model.early_exit2.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)
    else:
        raise ValueError("exit_number must be 1 or 2")

    train_args = {
        'train_function': trainEE_KL,
        'model': model,
        'id_dataloader': dataloaders_subset_a["train"],
        'id_testloader': dataloaders_subset_a["test"],
        'ood_dataloader': dataloaders_subset_b["train"],
        'exit_number': exit_number,
        'criterion_id': nn.CrossEntropyLoss(),
        'optimizer': optimizer,
        'stepsize': 30,
        'num_epochs': 5,
        'alpha': 1
    }

    results.update(get_system_status("Before Training", process))

    print(f"[SingleExit] 启动训练线程 Exit{exit_number} ...")
    train_timer = Timer(f"训练 Exit{exit_number} 耗时")
    train_timer.start()
    controller = TrainingController()
    controller.start_training(train_function=train_args.pop('train_function'), **train_args)
    controller.train_thread.join()
    train_timer.stop()

    results.update(get_system_status("After Training", process))
    results["Mode"] = f"Single Exit{exit_number} Training"
    results[f"Exit{exit_number} Training Time (s)"] = train_timer.elapsed()

    print(f"[SingleExit] Exit{exit_number} 训练完成")



def run_training_with_profiler(model, dataloaders_subset_a, ood_dataloaders, exit_number=1):
    import os
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


    log_name = f'prof_test_EE{exit_number}'
    os.makedirs(f'./log/{log_name}', exist_ok=True)    
    # ... setup optimizer, criterion etc. ...
        # === 冻结主干，解冻指定 early exit ===
    for param in model.parameters():
        param.requires_grad = False
    if exit_number == 1:
        for param in model.early_exit1.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.early_exit1.parameters(), lr=0.01)
    elif exit_number == 2:
        for param in model.early_exit2.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.early_exit2.parameters(), lr=0.01)
    else:
        raise ValueError("exit_number must be 1 or 2")
    
    criterion_id = nn.CrossEntropyLoss()
    model.to('cuda')

    # Use the profiler to wrap the call to our new function
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(f'./log/{log_name}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        print(f"[Profiler] Starting profiling for Exit{exit_number}...")
        
        trainEE_KL_for_profiling(
            model=model,
            id_dataloader=dataloaders_subset_a["train"],
            ood_dataloader=ood_dataloaders["train"],
            exit_number=exit_number,
            criterion_id=criterion_id,
            optimizer=optimizer,
            num_epochs=5,
            prof=prof,  # Pass the profiler object!
            profile_steps=10 # Ensure this is >= wait + warmup + active
        )



def run_testing_with_profiler(
    model,
    test_loader,
    exit_number=None,  # None for auto_select, 1/2/3 for forced exit    
    profile_steps=10
):
    """
    Profile the inference process for the given model and test_loader.
    Allows forcing a specific exit or using auto_select.
    """
    import os
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    import torch
    log_name = f'prof_test_EE{exit_number}'
    os.makedirs(f'./log/{log_name}', exist_ok=True)
    model.eval()
    model.to('cuda')

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(f'./log/{log_name}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        print(f"[Profiler] Starting inference profiling (exit={exit_number})...")
        step_count = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch[0].to('cuda'), batch[1].to('cuda')
                if exit_number == 1:
                    outputs, _ = model(images, return_exit1=True)
                elif exit_number == 2:
                    outputs, _ = model(images, return_exit2=True)
                elif exit_number == 3:
                    outputs, _ = model(images)
                else:
                    outputs, _ = model(images, auto_select=True)
                # (Optional) You can print or log outputs here
                prof.step()
                step_count += 1
                if step_count >= profile_steps:
                    print(f"[Profiler] Reached {profile_steps} steps. Stopping inference.")
                    break