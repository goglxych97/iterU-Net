import torch
import torch.nn as nn
from typing import Sequence, Union
from monai.utils import ensure_tuple_rep
from monai.networks.layers.factories import Conv
from .blocks import Down, UpCat, TwoConv
from .time_embedding import get_sinusoidal_positional_encoding, resize_and_project_time_embedding
from .ConvLSTMCell import ConvLSTMCell3D


class iterUNet_wm(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        num_iterations: int = 4,  # 0,1->2 #0,2->3 #0,3->4 #0,4->5
        time_embedding_dim: int = 128,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        
        self.num_iterations = num_iterations
        self.time_embedding_dim = time_embedding_dim
        self.time_embeddings = [
            get_sinusoidal_positional_encoding(
                torch.tensor([t]), self.time_embedding_dim) for t in range(num_iterations)
        ]
        
        self.mtp0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.mtp1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.mtp2 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.mtp3 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.mtp4 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        
        self.time_projection0 = nn.Linear(time_embedding_dim, fea[0])
        self.time_projection1 = nn.Linear(time_embedding_dim, fea[1])
        self.time_projection2 = nn.Linear(time_embedding_dim, fea[2])
        self.time_projection3 = nn.Linear(time_embedding_dim, fea[3])
        self.time_projection4 = nn.Linear(time_embedding_dim, fea[4])
        
        self.conv_lstm = ConvLSTMCell3D(fea[0], fea[0])
            
        # Encoder
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        
        # Decoder
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
       
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
        if self.training:
            x_pre = x[:, 0, :, :, :].unsqueeze(1)
            iter_outputs = []
            hidden_state = None
        
            for i in range(self.num_iterations):
                x_post = x[:, i+1, :, :, :].unsqueeze(1)
                time_emb = self.time_embeddings[i].to(x.device)
            
                d0 = self.conv_0(torch.cat((x_pre, x_post,), axis=1))
                hidden_state, cell_state = self.conv_lstm(d0, (hidden_state, cell_state) if hidden_state is not None else None)
                d0 = hidden_state
            
                d0_emd = resize_and_project_time_embedding(self.mtp0(time_emb), d0.shape[2:], self.time_projection0)
                d0 = d0+d0_emd
                d1 = self.down_1(d0)
                d1_emd = resize_and_project_time_embedding(self.mtp1(time_emb), d1.shape[2:], self.time_projection1)
                d1 = d1+d1_emd
                d2 = self.down_2(d1)
                d2_emd = resize_and_project_time_embedding(self.mtp2(time_emb), d2.shape[2:], self.time_projection2)
                d2 = d2+d2_emd   
                d3 = self.down_3(d2)
                d3_emd = resize_and_project_time_embedding(self.mtp3(time_emb), d3.shape[2:], self.time_projection3)
                d3 = d3+d3_emd    
                d4 = self.down_4(d3)
                d4_emd = resize_and_project_time_embedding(self.mtp4(time_emb), d4.shape[2:], self.time_projection4)
                d4 = d4+d4_emd              
            
                u4 = self.upcat_4(d4, d3)
                u4_emd = resize_and_project_time_embedding(self.mtp3(time_emb), u4.shape[2:], self.time_projection3)
                u4 = u4+u4_emd 
                u3 = self.upcat_3(u4, d2)
                u3_emd = resize_and_project_time_embedding(self.mtp2(time_emb), u3.shape[2:], self.time_projection2)
                u3 = u3+u3_emd       
                u2 = self.upcat_2(u3, d1)
                u2_emd = resize_and_project_time_embedding(self.mtp1(time_emb), u2.shape[2:], self.time_projection1)
                u2 = u2+u2_emd
                u1 = self.upcat_1(u2, d0)
                u1_emd = resize_and_project_time_embedding(self.mtp0(time_emb), u1.shape[2:], self.time_projection0)
                u1 = u1+u1_emd
            
                iter_output = self.final_conv(u1) + x_post
                iter_outputs.append(iter_output)
    
            return tuple(iter_outputs)
            
        else:
            x_pre = x[:, 0, :, :, :].unsqueeze(1)
            iter_outputs = []
            hidden_state = None
        
            for i in range(self.num_iterations):
                if i == 0:
                    x_post = x[:, i+1, :, :, :].unsqueeze(1)
                else:
                    x_post = iter_outputs[-1]
                time_emb = self.time_embeddings[i].to(x.device)
            
                d0 = self.conv_0(torch.cat((x_pre, x_post,), axis=1))
                hidden_state, cell_state = self.conv_lstm(d0, (hidden_state, cell_state) if hidden_state is not None else None)
                d0 = hidden_state
            
                d0_emd = resize_and_project_time_embedding(self.mtp0(time_emb), d0.shape[2:], self.time_projection0)
                d0 = d0+d0_emd
                d1 = self.down_1(d0)
                d1_emd = resize_and_project_time_embedding(self.mtp1(time_emb), d1.shape[2:], self.time_projection1)
                d1 = d1+d1_emd
                d2 = self.down_2(d1)
                d2_emd = resize_and_project_time_embedding(self.mtp2(time_emb), d2.shape[2:], self.time_projection2)
                d2 = d2+d2_emd   
                d3 = self.down_3(d2)
                d3_emd = resize_and_project_time_embedding(self.mtp3(time_emb), d3.shape[2:], self.time_projection3)
                d3 = d3+d3_emd    
                d4 = self.down_4(d3)
                d4_emd = resize_and_project_time_embedding(self.mtp4(time_emb), d4.shape[2:], self.time_projection4)
                d4 = d4+d4_emd              
            
                u4 = self.upcat_4(d4, d3)
                u4_emd = resize_and_project_time_embedding(self.mtp3(time_emb), u4.shape[2:], self.time_projection3)
                u4 = u4+u4_emd 
                u3 = self.upcat_3(u4, d2)
                u3_emd = resize_and_project_time_embedding(self.mtp2(time_emb), u3.shape[2:], self.time_projection2)
                u3 = u3+u3_emd       
                u2 = self.upcat_2(u3, d1)
                u2_emd = resize_and_project_time_embedding(self.mtp1(time_emb), u2.shape[2:], self.time_projection1)
                u2 = u2+u2_emd
                u1 = self.upcat_1(u2, d0)
                u1_emd = resize_and_project_time_embedding(self.mtp0(time_emb), u1.shape[2:], self.time_projection0)
                u1 = u1+u1_emd
            
                iter_output = self.final_conv(u1) + x_post
                iter_outputs.append(iter_output)
    
            return tuple(iter_outputs)
