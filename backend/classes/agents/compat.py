import sys
import torch


print("Python Version:", sys.version)
print("Python Architecture:", "64-bit" if sys.maxsize > 2**32 else "32-bit")
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Compilation Support:", torch.cuda.is_available() and torch.__version__ >= "2.0")
