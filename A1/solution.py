import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from scipy.optimize import minimize
from scipy import stats
import numpy as np

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
#                      ! DO NOT MODIFY THESE FUNCTIONS !
def integer_linear(input, weight):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.linear(input, weight)
    else:
        output = F.linear(input.float(), weight.float())
        output = output.round().to(torch.int32)
    return output
def integer_conv2d(input, weight, stride, padding, dilation, groups):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
    else:
        output = F.conv2d(input.float(), weight.float(), None, stride, padding, dilation, groups)
        output = output.round().to(torch.int32)
    return output
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def linear_quantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                    N_bits: int, signed: bool = True) -> torch.Tensor:
    """
    linear uniform quantization for real tensor
    Args:
        input: torch.tensor
        scale: scale factor
        zero_point: zero point
        N_bits: bitwidth
        signed: flag to indicate signed ot unsigned quantization

    Returns:
        quantized_tensor: quantized tensor whose values are integers
    """

    ##### WRITE CODE HERE #####
    quantized_tensor = torch.zeros_like(input)
    return quantized_tensor

def linear_dequantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    linear uniform de-quantization for quantized tensor
    Args:
        input: input quantized tensor
        scale: scale factor
        zero_point: zero point

    Returns:
        reconstructed_tensor: de-quantized tensor whose values are real
    """
    ##### WRITE CODE HERE #####
    quantized_tensor = torch.zeros_like(input)
    return quantized_tensor


def get_scale(input, N_bits=8):
    """
    extract optimal scale based on statistics of the input tensor.
    Args:
        input: input real tensor
        N_bits: bitwidth
    Returns:
        scale optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.032, 1.085]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    ##### WRITE CODE HERE #####
    q_scale = 1.0
    return q_scale


def reset_scale_and_zero_point(input: torch.tensor, N_bits: int = 4, method: str = "sym"):
    """
    Args:
        input: input real tensor
        N_bits: bitwidth
        method: choose between sym, unsigned, SAWB, and heuristic
    Returns:
        scale factor , zero point
    """
    with torch.no_grad():
        if method == 'heuristic':
            ##### WRITE CODE HERE #####
            # step_size = argmin_{step_size} (MSE[x, x_hat])
            zero_point = torch.tensor(0.)
            step_size = torch.tensor(1.)
        elif method == 'SAWB':
            ##### WRITE CODE HERE #####
            zero_point = torch.tensor(0.)
            step_size = torch.tensor(1.)
        elif method == 'sym':
            ##### WRITE CODE HERE #####
            zero_point = torch.tensor(0.)
            step_size = torch.tensor(1.)
        elif method == 'unsigned':
            ##### WRITE CODE HERE #####
            zero_point = torch.tensor(0.)
            step_size = torch.tensor(1.)
        else:
            raise "didn't find quantization method."

    return step_size, zero_point


class _quantize_func_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, N_bits, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: torch.tensor
            scale: scale factor
            zero_point: zero point
            N_bits: bitwidth
            signed: flag to indicate signed ot unsigned quantization
        Returns:
            quantized_tensor: quantized tensor whose values are integers
        """
        ctx.scale = scale
        ##### WRITE CODE HERE #####
        quantized_tensor = torch.zeros_like(input)
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        ##### WRITE CODE HERE #####
        grad_input = grad_output
        return grad_input, None, None, None, None

linear_quantize_STE = _quantize_func_STE.apply


def quantized_linear_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float):
    """
    integer only fully connected layer. 
    Note that you are only allowed to use <integer_linear> function!
    Args:
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    ##### WRITE CODE HERE #####
    output = torch.zeros_like(input)
    return output


def quantized_conv2d_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float, stride,
                              padding, dilation, groups):
    """
    integer only fully connected layer
    Note that you are only allowed to use <integer_conv2d> function!
    Args:
        groups: number of groups
        stride: stride
        dilation: dilation
        padding: padding
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    ##### WRITE CODE HERE #####
    output = torch.zeros_like(input)
    return output


class Quantized_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.linear(input, self.weight, self.bias)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, False)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_linear_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.linear(input_reconstructed, weight_reconstructed, self.bias)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_and_zero_point(input, self.act_N_bits, 'unsigned')
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _= reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, False)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_conv2d_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale, self.stride,
                                               self.padding, self.dilation, self.groups)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.conv2d(input_reconstructed, weight_reconstructed, self.bias, self.stride, self.padding,
                                        self.dilation, self.groups)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_and_zero_point(input, self.act_N_bits, 'unsigned')
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _ = reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)
