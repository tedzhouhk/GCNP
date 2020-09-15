import numpy as np

import torch
from torch.autograd import Variable

import gc

def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def print_allocated_tensors():
    print('>>>>> allocated tensors <<<<<')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass