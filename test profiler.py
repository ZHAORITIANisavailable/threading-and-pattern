import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import os

# 设置日志目录
log_dir = "./log/trace_new"
print("Log files will be saved to:", os.path.abspath(log_dir))

# 简单模型
model = torch.nn.Linear(10, 10).cuda()
x = torch.randn(5, 10).cuda()

# 使用 profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
    on_trace_ready=tensorboard_trace_handler(log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(4):
        y = model(x)
        prof.step()

##tensorboard --logdir=./log/trace_new
