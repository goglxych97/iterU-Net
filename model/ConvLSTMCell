import torch
import torch.nn as nn

class ConvLSTMCell3D(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell3D, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels + hidden_channels,  
            hidden_channels * 4,  
            kernel_size,
            padding=padding
        )

    def init_hidden(self, x):
        b, _, d, h, w = x.shape
        return (
            torch.zeros(b, self.hidden_channels, d, h, w, device=x.device),
            torch.zeros(b, self.hidden_channels, d, h, w, device=x.device),
        )

    def forward(self, x, prev_state=None):
        if prev_state is None:
            prev_state = self.init_hidden(x)

        h_prev, c_prev = prev_state
        combined = torch.cat([x, h_prev], dim=1)  # 입력과 이전 hidden state 결합
        gates = self.conv(combined)
        
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
