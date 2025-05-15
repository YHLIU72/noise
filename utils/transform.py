import numpy as np
import torch

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def __call__(self, x):
        x = x.clone().detach().to(torch.float32)
        return (x - self.mean) / self.std

class NumpyNormalizer(object):
    def __init__(self, mean, std):
        # 将均值和标准差转换为 numpy 数组
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, x):
        # 确保输入是 numpy 数组
        x = np.array(x, dtype=np.float32)
        return (x - self.mean) / self.std
        
