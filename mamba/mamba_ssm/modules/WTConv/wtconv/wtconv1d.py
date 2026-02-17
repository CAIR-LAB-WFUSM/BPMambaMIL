import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .util import wavelet
import pywt
class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.kernel_size = kernel_size
        self.wt_type = wt_type

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', stride=1, bias=bias)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels, bias=False)
            for _ in range(wt_levels)
        ])
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1))

        # 创建小波滤波器
        wavelet = pywt.Wavelet(wt_type)
        self.dec_lo = nn.Parameter(torch.Tensor(wavelet.dec_lo[::-1]), requires_grad=False)
        self.dec_hi = nn.Parameter(torch.Tensor(wavelet.dec_hi[::-1]), requires_grad=False)
        self.rec_lo = nn.Parameter(torch.Tensor(wavelet.rec_lo), requires_grad=False)
        self.rec_hi = nn.Parameter(torch.Tensor(wavelet.rec_hi), requires_grad=False)

    def wavedec(self, x):
        coeffs = []
        for _ in range(self.wt_levels):
            x, d = self.dwt(x)
            coeffs.append(d)
        coeffs.append(x)
        return coeffs[::-1]

    def waverec(self, coeffs):
        x = coeffs[0]
        for d in coeffs[1:]:
            x = self.idwt(x, d)
        return x

    def dwt(self, x):
        lo = F.conv1d(x, self.dec_lo.view(1, 1, -1).expand(self.in_channels, -1, -1), 
                      groups=self.in_channels, padding='same')
        hi = F.conv1d(x, self.dec_hi.view(1, 1, -1).expand(self.in_channels, -1, -1), 
                      groups=self.in_channels, padding='same')
        return lo[:, :, ::2], hi[:, :, ::2]

    def idwt(self, lo, hi):
        lo = F.interpolate(lo, scale_factor=2, mode='nearest')
        hi = F.interpolate(hi, scale_factor=2, mode='nearest')
        lo = F.conv1d(lo, self.rec_lo.view(1, 1, -1).expand(self.in_channels, -1, -1), 
                      groups=self.in_channels, padding='same')
        hi = F.conv1d(hi, self.rec_hi.view(1, 1, -1).expand(self.in_channels, -1, -1), 
                      groups=self.in_channels, padding='same')
        
        # 确保 lo 和 hi 具有相同的长度
        min_len = min(lo.size(2), hi.size(2))
        lo = lo[:, :, :min_len]
        hi = hi[:, :, :min_len]
        
        return lo + hi

    def forward(self, x):
        # 确保输入是3D (B, C, L)
        if x.dim() == 4:
            x = x.squeeze(2)
        elif x.dim() != 3:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        # 保存原始长度
        orig_len = x.size(2)

        # 应用基础卷积
        out = self.conv(x)

        # 应用小波变换
        coeffs = self.wavedec(x)
        
        for i, (coeff, conv) in enumerate(zip(coeffs[1:], self.wavelet_convs)):
            transformed = conv(coeff)
            coeffs[i+1] = transformed

        # 重构信号
        reconstructed = self.waverec(coeffs)

        # 确保重构信号与原始信号长度相同
        if reconstructed.size(2) != orig_len:
            reconstructed = F.interpolate(reconstructed, size=orig_len, mode='linear', align_corners=False)

        # 应用缩放并添加到输出
        out = out + self.scale * reconstructed

        if self.stride > 1:
            out = out[:, :, ::self.stride]

        return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    
    def forward(self, x):
        return torch.mul(self.weight, x)

