import torch
# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available}")
# 打印CUDA版本
cuda_version = torch.version.cuda
print(f"CUDA版本: {cuda_version}")
# 检查cuDNN是否可用
cudnn_available = torch.backends.cudnn.is_available()
print(f"cuDNN是否可用: {cudnn_available}")
# 打印cuDNN版本
cudnn_version = torch.backends.cudnn.version()
print(f"cuDNN版本: {cudnn_version}")