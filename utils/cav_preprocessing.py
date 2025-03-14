import torch
import numpy as np

def build_numpy_list_cav(data:torch.Tensor) -> list:

    data = data.detach().cpu()

    data_np = [
            data[i].numpy()[-5:].flatten()
            for i in range(len(data))
        ]
    
    
    
    return data_np