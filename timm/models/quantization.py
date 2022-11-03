import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

DTYPE = torch.int8
QMAX = 127
QMIN = -128
THRESHOLD = 5e-7

def get_quantization_scale_and_zero_point_from_range(fp_min, fp_max):
    scale = (fp_max - fp_min) / (QMAX - QMIN)
    zero_point = int(round(QMIN - fp_min / scale))
    if zero_point < QMIN:
        zero_point = QMIN
    elif zero_point > QMAX:
        zero_point = QMAX
    return scale, zero_point

def get_quantization_scale_and_zero_point(fp_tensor):
    return get_quantization_scale_and_zero_point_from_range(fp_tensor.min().item(), fp_tensor.max().item())

def get_quantization_scale_and_zero_point_per_head(fp_tensor):
    fp_max = fp_tensor.amax(dim=(0, 2, 3), keepdim=True)
    fp_min = fp_tensor.amin(dim=(0, 2, 3), keepdim=True)
    scale = (fp_max - fp_min) / (QMAX - QMIN)
    zero_point = (QMIN - fp_min / scale).round().clamp(min=QMIN, max=QMAX).to(DTYPE)
    return scale, zero_point

def _smooth_distribution(p, eps=0.0001):
    is_zeros = (p == 0).to(torch.float32)
    is_nonzeros = (p != 0).to(torch.float32)
    n_zeros = is_zeros.sum().item()
    n_nonzeros = len(p) - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.to(torch.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum().item() == 0
    return hist

def kl_divergence_scale(arr, num_bins=8001):
    num_quantized_bins = QMAX - QMIN
    min_val = arr.min().item()
    max_val = arr.max().item()
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = torch.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2

    thresholds = torch.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = torch.zeros_like(thresholds)
    quantized_bins = torch.zeros(num_quantized_bins)
    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        p = sliced_nd_hist.clone()
        assert len(p) % 2 == 1
        assert len(p) >= num_quantized_bins
        left_outlier_count = hist[0:p_bin_idx_start].sum().item()
        p[0] += left_outlier_count
        right_outlier_count = hist[p_bin_idx_stop:].sum().item()
        p[-1] += right_outlier_count
        is_nonzeros = (p != 0).to(torch.int32)

        num_merged_bins = len(sliced_nd_hist) // num_quantized_bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum().item()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum().item()

        q = torch.zeros(sliced_nd_hist.shape, dtype=torch.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum().item()
            if norm != 0:
                q[start:stop] = quantized_bins[j] / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = F.kl_div(q, p)

    min_divergence_idx = torch.argmin(divergence)
    return thresholds[min_divergence_idx].item()

def i_exp(q, scale):
    def i_poly(x, scale):  # ax^2+bx+c
        a, b, c = 0.35815147, 0.96963238, 1
        q_b = round(b / a / scale)
        q_c = round(c / a / scale ** 2)
        return x ** 2 + q_b * x + q_c, a * scale ** 2

    q_ln2 = round(log(2) / scale)
    z = (-q / q_ln2).round()
    q_p = q + z * q_ln2
    q_l, s_l = i_poly(q_p, scale)
    return q_l / 2 ** z, s_l

def i_softmax(q, scale):
    q = q - q.max()
    q_exp, exp_scale = i_exp(q, scale)
    return (q_exp / q_exp.sum(dim=-1, keepdim=True)).round(), exp_scale

class ActivationInfo:
    def __init__(self):
        self.min = float("inf")
        self.max = float("-inf")
        self.activations = torch.empty(0)
        self.kl_threshold = None
    
    def update(self, x):
        self.min = min(self.min, x.min().item())
        self.max = max(self.max, x.max().item())
        self.activations = torch.concat((self.activations, x.reshape(-1).cpu()), 0)
    
    def calc_threshold(self, kl=True):
        if kl:
            self.kl_threshold = kl_divergence_scale(self.activations, num_bins=2001)
        else:
            self.kl_threshold = max(self.max, -self.min)
        self.max = self.kl_threshold
        self.min = max(self.min, -self.kl_threshold)

class QuantizedConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, act_quant=True, input_act_range=None):
        super().__init__()
        self.act_quant = act_quant
        # scale = max(module.weight.data.abs().max().item(), THRESHOLD) / QMAX
        self.weight_scale = conv.weight.data.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=THRESHOLD) / QMAX
        self.weight = (conv.weight.data / self.weight_scale).round()
        self.bias = None if conv.bias is None else conv.bias.data.reshape(1, -1, 1, 1)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.input_act_range = input_act_range
    
    def forward(self, x):
        if not self.act_quant:
            x = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            x = x * self.weight_scale.transpose(0, 1)
        else:
            if self.input_act_range:
                scale, zero_point = get_quantization_scale_and_zero_point_from_range(*self.input_act_range)
            else:
                scale, zero_point = get_quantization_scale_and_zero_point(x)
            q_input = (x / scale + zero_point).round()
            x = F.conv2d(q_input - zero_point, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            x = x * scale * self.weight_scale.transpose(0, 1)
        if self.bias is not None:
            x += self.bias
        return x

class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, act_quant=True, input_act_range=None):
        super().__init__()
        self.act_quant = act_quant
        # self.raw_linear = linear
        self.weight_scale = linear.weight.data.abs().amax(dim=1, keepdim=True).clamp(min=THRESHOLD) / QMAX
        # self.weight_scale = max(linear.weight.data.abs().max().item(), THRESHOLD) / QMAX
        self.weight = (linear.weight / self.weight_scale).round()
        self.bias = linear.bias
        self.weight_scale = self.weight_scale.reshape(1, 1, -1)
        self.input_act_range = input_act_range
    
    def forward(self, x):
        if len(self.weight_scale.shape) != len(x.shape):
            self.weight_scale = self.weight_scale.squeeze(0)
        # raw_result = self.raw_linear(x)
        if not self.act_quant:
            x = F.linear(x, self.weight, None) * self.weight_scale
        else:
            if self.input_act_range:
                scale, zero_point = get_quantization_scale_and_zero_point_from_range(*self.input_act_range)
            else:
                scale, zero_point = get_quantization_scale_and_zero_point(x)
            q_input = (x / scale + zero_point).round()
            x = F.linear(q_input - zero_point, self.weight, None)
            x = x * scale * self.weight_scale
        if self.bias is not None:
            x += self.bias
        # print((raw_result - x).abs().mean())
        return x

if __name__ == "main":
    x = torch.rand(1000, 1000)
    scale = x.abs().max().item() / QMAX
    q = (x / scale).round()
    q -= q.max().item()
    real_softmax = (q * scale).softmax(dim=-1)
    q_softmax, softmax_scale = i_softmax(q, scale)
    print((real_softmax - q_softmax * softmax_scale).abs().mean().item())
