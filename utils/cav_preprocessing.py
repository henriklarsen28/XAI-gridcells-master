import torch
import numpy as np

def build_numpy_list_cav(data:torch.Tensor) -> list:

    data = data.detach().cpu()

    data_np = [
            data[i].numpy()[-2:].flatten()
            for i in range(len(data))
        ]
    
    #print("data_np", data_np[0].shape) 
    
    return data_np