import psutil
import torch

def print_resource_usage():
    """Print current system resource usage statistics."""
    # GPU memory usage
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_used = mem.used / (1024 ** 2)
    mem_total = mem.total / (1024 ** 2)

    print(f"[Resource Monitor] GPU Allocated: {allocated:.1f} MB | GPU Reserved: {reserved:.1f} MB | "
          f"CPU Usage: {cpu_percent:.1f}% | Memory Usage: {mem_used:.1f}/{mem_total:.1f} MB")
