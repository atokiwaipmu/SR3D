
import torch
import numpy as np

def grid_chunk(x, flag=False):
    if len(x.shape) == 4:
        x = x.unsqueeze(0)
        flag = True
    batch, channel, height, width, depth = x.shape
    subbox = [x[:,:,i*height//2:(i+1)*height//2,j*width//2:(j+1)*width//2,k*depth//2:(k+1)*depth//2] 
            for i in range(2) for j in range(2) for k in range(2)]
    if flag:
        subbox = [subbox[i].squeeze(0) for i in range(8)]
    return subbox

def grid_unchunk(subbox, flag=False):
    if len(subbox[0].shape) == 4:
        subbox = [subbox[i].unsqueeze(0) for i in range(8)]
        flag = True
    batch, channel, height, width, depth = subbox[0].shape
    x = torch.zeros((batch, channel, height*2, width*2, depth*2), device=subbox[0].device)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x[:,:,i*height:(i+1)*height,j*width:(j+1)*width,k*depth:(k+1)*depth] = subbox[i*4 + j*2 + k]
    
    if flag:
        x = x.squeeze(0)
    return x

def get_masked_grid(x, mask=None, x_t=None):
    subbox = grid_chunk(x)
    if mask is None:
        mask = np.ones(8)
    if x_t is None:
        x_t = torch.randn_like(x)
    x_t = grid_chunk(x_t)
    x = [x_t[i] if mask[i] else subbox[i] for i in range(8)]
    x = grid_unchunk(x)
    return x

def extract_masked_region(x, mask):
    # get the chunks where mask is 1
    x = grid_chunk(x)
    x = [x[i] for i in range(8) if mask[i]]
    x = torch.cat(x, dim=0)
    return x