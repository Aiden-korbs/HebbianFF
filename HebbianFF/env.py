import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_LOGS", "-graph_breaks")
os.environ.setdefault("TORCH_COMPILE_DISABLE_SIZE_LIMIT", "1")
