import math
import torch
import torch.nn as nn

def get_sinusoidal_positional_encoding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    return emb

def resize_and_project_time_embedding(time_emb, target_shape, projection_layer):
    time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, time_dim, 1, 1, 1)
    time_emb = time_emb.expand(-1, -1, target_shape[0], target_shape[1], target_shape[2])  # 공간 차원 확장
    time_emb = time_emb.permute(0, 2, 3, 4, 1)  # (batch_size, depth, height, width, time_dim)
    time_emb = projection_layer(nn.GELU()(time_emb))
    time_emb = time_emb.permute(0, 4, 1, 2, 3)  # (batch_size, target_channels, depth, height, width)
    
    return time_emb
