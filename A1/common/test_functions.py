import torch

from solution import *
from scipy.optimize import minimize
from scipy import stats
import numpy as np
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict


def test_linear_quantize():
    test_tensor = torch.tensor([
        [0.0523, 0.6364, -0.0968, -0.0020, 0.1940],
        [0.7500, 0.5507, 0.6188, -0.1734, 0.4677],
        [-0.0669, 0.3836, 0.4297, 0.6267, -0.0695],
        [0.1536, -0.0038, 0.6075, 0.6817, 0.0601],
        [0.6446, -0.2500, 0.5376, -0.2226, 0.2333]])
    quantized_test_tensor = torch.tensor([
        [-1, 1, -1, -1, 0],
        [1, 1, 1, -2, 0],
        [-1, 0, 0, 1, -1],
        [-1, -1, 1, 1, -1],
        [1, -2, 1, -2, 0]], dtype=torch.int8)
    real_min = -0.25
    real_max = 0.75
    bitwidth = 2
    scale = torch.tensor(1 / 3)
    zero_point = torch.tensor(1)
    _quantized_test_tensor = linear_quantize(test_tensor, scale, zero_point, bitwidth).to(
        quantized_test_tensor.dtype)
    assert _quantized_test_tensor.equal(quantized_test_tensor)
    print('* Test passed.')


def test_linear_dequantize():
    test_tensor = torch.tensor([
        [0.0523, 0.6364, -0.0968, -0.0020, 0.1940],
        [0.7500, 0.5507, 0.6188, -0.1734, 0.4677],
        [-0.0669, 0.3836, 0.4297, 0.6267, -0.0695],
        [0.1536, -0.0038, 0.6075, 0.6817, 0.0601],
        [0.6446, -0.2500, 0.5376, -0.2226, 0.2333]])
    quantized_test_tensor = torch.tensor([
        [-1, 1, -1, -1, 0],
        [1, 1, 1, -2, 0],
        [-1, 0, 0, 1, -1],
        [-1, -1, 1, 1, -1],
        [1, -2, 1, -2, 0]], dtype=torch.int8)
    real_min = -0.25
    real_max = 0.75
    bitwidth = 2
    scale = torch.tensor(1 / 3)
    zero_point = torch.tensor(1)
    _quantized_test_tensor = linear_quantize(test_tensor, scale, zero_point, bitwidth).to(
        quantized_test_tensor.dtype)
    test_reconstructed = linear_dequantize(_quantized_test_tensor, scale, zero_point)
    assert torch.allclose(test_tensor, test_reconstructed, atol=scale)
    print('* Test passed.')


def generate_samples(distribution, num_samples=1000000):
    if distribution == 'normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif distribution == 'uniform':
        dist = torch.distributions.uniform.Uniform(-1, 1)
    elif distribution == 'laplacian':
        dist = torch.distributions.laplace.Laplace(0, 1)
    elif distribution == 'gamma':
        dist = torch.distributions.gamma.Gamma(torch.tensor([1.]), torch.tensor([2.0]))
    elif distribution == 'von_mises':
        dist = torch.distributions.von_mises.VonMises(torch.tensor([0.]), torch.tensor([4.0]))
    else:
        raise "didn't find distribution"

    x = dist.sample((num_samples,))
    return x


def test_update_scale_and_zero_point():
    num_samples = 1000000
    distinct_distributions_list = ['normal', 'uniform', 'laplacian', 'von_mises']
    methods = ['sym', 'SAWB']
    bits = [8, 4, 2]
    for method in methods:
        for b in bits:
            for dist in distinct_distributions_list:
                test_tensor = generate_samples(dist, num_samples)
                scale, zero_point = reset_scale_and_zero_point(test_tensor, b, method)
                quantized_tensor = linear_quantize(test_tensor, scale, zero_point, b)
                dequantized_tensor = linear_dequantize(quantized_tensor, scale, zero_point)
                mse = torch.nn.functional.mse_loss(test_tensor, dequantized_tensor)/test_tensor.abs().max()
                atol = test_tensor.abs().max()/(2**(b-1))
                assert torch.allclose(test_tensor, dequantized_tensor, atol=atol) or mse<0.05 , f"fiald for {b}bit quantization with {method} method!"
    print('* Test passed.')

def test_STE_grad():
    N_bits=8
    method='sym'
    real_weight = torch.rand(10, 10).requires_grad_(True)
    weight_scale, zero_point = reset_scale_and_zero_point(real_weight, N_bits, method)
    quantized_weight = linear_quantize_STE(real_weight, weight_scale, zero_point, N_bits, True)
    reconstructed_weight = linear_dequantize(quantized_weight, weight_scale, zero_point)
    grad = torch.autograd.grad(reconstructed_weight.sum(), real_weight)[0]
    assert (grad == torch.tensor(1.)).sum() == grad.numel()
    print('* Test passed.')
    

def test_quantized_linear_function(weight_N_bits=4, act_N_bits=4, method='sym', bias=True):
    m = torch.nn.Linear(1000, 100, bias)
    real_weight = m.weight.detach()
    if bias:
        real_bias = m.bias.detach()
    else:
        real_bias = None
    real_input = torch.rand(128, 1000)
    real_output = m(real_input).detach()

    weight_scale, zero_point = reset_scale_and_zero_point(real_weight, weight_N_bits, method)
    input_scale, zero_point = reset_scale_and_zero_point(real_input, act_N_bits, 'unsigned')

    quantized_weight = linear_quantize_STE(real_weight, weight_scale, zero_point, weight_N_bits, True).to(torch.int32)
    quantized_input = linear_quantize_STE(real_input, input_scale, zero_point, act_N_bits, False).to(torch.int32)
    if real_bias is None:
        quantized_bias = None
    else:
        quantized_bias = linear_quantize_STE(real_bias, weight_scale * input_scale, zero_point, 32).to(torch.int32)

    reconstructed_input = linear_dequantize(quantized_input, input_scale, zero_point)
    reconstructed_weight = linear_dequantize(quantized_weight, weight_scale, zero_point)

    print("=======================input tensor=======================")
    print("-> maximum abs error in quantizing input")
    print(f'    expected based on scale: {input_scale / 2} ')
    print(f'    evaluated: {abs(real_input - reconstructed_input).max()} ')
    assert torch.allclose(real_input, reconstructed_input, atol=input_scale / 2)
    print("=======================weight tensor=======================")
    print("-> maximum abs error in quantizing weight")
    print(f'    expected based on scale: {weight_scale / 2} ')
    print(f'    evaluated: {abs(real_weight - reconstructed_weight).max()} ')
    assert torch.allclose(real_weight, reconstructed_weight, atol=weight_scale / 2) or torch.nn.functional.mse_loss(
        real_weight, reconstructed_weight).item() / real_weight.abs().max() < 0.01

    simulated_output = simulated_output = F.linear(reconstructed_input, reconstructed_weight, real_bias)
    output = quantized_linear_function(quantized_input, quantized_weight, quantized_bias, input_scale, weight_scale)
    atol = max(real_input.sum(1).max() * weight_scale, real_weight.sum(1).max() * input_scale)
    print("==============real_output -vs- simulated_output=============")
    print("-> maximum abs error between real_output and simulated output")
    print(f'    expected based on scale: {atol} ')
    print(f'    evaluated: {abs(real_output - simulated_output).max()} ')
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, simulated_output)} ')
    assert torch.allclose(real_output, simulated_output, atol=atol)
    print("================real_output -vs- int_output=================")
    print("-> maximum abs error between real_output and integer only output")
    print(f'    expected based on scale: {atol} ')
    print(f'    evaluated: {abs(real_output - output).max()} ')
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, output)} ')
    assert torch.allclose(real_output, output, atol=atol)   
    print('* Test passed.')


def test_quantized_linear_module(weight_N_bits=8, act_N_bits=8, method='sym', bias=True):
    m_q = Quantized_Linear(1000, 100, bias)
    m_q.method = method
    m_q.weight_N_bits = weight_N_bits
    m_q.act_N_bits = act_N_bits

    real_input = torch.rand(128, 1000)
    m_q.input_scale.data, _ = reset_scale_and_zero_point(real_input, act_N_bits, 'unsigned')
    m_q.method = 'normal'
    real_output = m_q(real_input)
    real_gard = torch.autograd.grad(real_output.sum(), m_q.weight)[0]
    m_q.method = method
    quntized_output = m_q(real_input)
    quntized_gard = torch.autograd.grad(quntized_output.sum(), m_q.weight)[0]

    print("=================real_output -vs- int_output=================")
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, quntized_output)} ')
    print("=============grad(real_output), grad(int_output)=============")
    print(f'    Abs-Mean: {(quntized_gard-real_gard).abs().mean()} ')
    #assert (quntized_gard-real_gard).abs().mean() < 5e-2
    #print('* Test passed.')


def test_quantized_conv2d_function(weight_N_bits=4, act_N_bits=4, method='sym', bias=True):
    m = torch.nn.Conv2d(64, 64, 3, bias=bias)
    real_weight = m.weight.detach()
    if bias:
        real_bias = m.bias.detach()
    else:
        real_bias = None
    real_input = torch.rand(128, 64, 32, 32)
    real_output = m(real_input).detach()

    weight_scale, zero_point = reset_scale_and_zero_point(real_weight, weight_N_bits, method)
    input_scale, zero_point = reset_scale_and_zero_point(real_input, act_N_bits, 'unsigned')

    quantized_weight = linear_quantize_STE(real_weight, weight_scale, zero_point, weight_N_bits, True).to(torch.int32)
    quantized_input = linear_quantize_STE(real_input, input_scale, zero_point, act_N_bits, False).to(torch.int32)
    if real_bias is None:
        quantized_bias = None
    else:
        quantized_bias = linear_quantize_STE(real_bias, weight_scale * input_scale, zero_point, 32).to(torch.int32)

    reconstructed_input = linear_dequantize(quantized_input, input_scale, zero_point)
    reconstructed_weight = linear_dequantize(quantized_weight, weight_scale, zero_point)

    print("=======================input tensor=======================")
    print("-> maximum abs error in quantizing input")
    print(f'    expected based on scale: {input_scale / 2} ')
    print(f'    evaluated: {abs(real_input - reconstructed_input).max()} ')
    assert torch.allclose(real_input, reconstructed_input, atol=input_scale / 2)
    print("=======================weight tensor=======================")
    print("-> maximum abs error in quantizing weight")
    print(f'    expected based on scale: {weight_scale / 2} ')
    print(f'    evaluated: {abs(real_weight - reconstructed_weight).max()} ')
    assert torch.allclose(real_weight, reconstructed_weight, atol=weight_scale / 2) or torch.nn.functional.mse_loss(
        real_weight, reconstructed_weight).item() / real_weight.abs().max() < 0.01

    simulated_output = F.conv2d(reconstructed_input, reconstructed_weight, m.bias, m.stride,
                                m.padding, m.dilation, m.groups)
    output = quantized_conv2d_function(quantized_input, quantized_weight, quantized_bias, input_scale,
                                       weight_scale, m.stride, m.padding, m.dilation, m.groups)
    atol = max(real_input.sum((2, 3)).max() * weight_scale, real_weight.sum((2, 3)).max() * input_scale)
    print("================real_output -vs- simulated_output================")
    print("-> maximum abs error between real_output and simulated output")
    print(f'    expected based on scale: {atol} ')
    print(f'    evaluated: {abs(real_output - simulated_output).max()} ')
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, simulated_output)} ')
    assert torch.allclose(real_output, simulated_output, atol=atol)
    print("=================real_output -vs- int_output=================")
    print("-> maximum abs error between real_output and integer only output")
    print(f'    expected based on scale: {atol} ')
    print(f'    evaluated: {abs(real_output - output).max()} ')
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, output)} ')
    assert torch.allclose(real_output, output, atol=atol)
    print('* Test passed.')


def test_quantized_conv2d_module(weight_N_bits=8, act_N_bits=8, method='sym', bias=True):
    m_q = Quantized_Conv2d(64, 64, 3, bias=bias)
    m_q.weight_N_bits = weight_N_bits
    m_q.act_N_bits = act_N_bits

    real_input = torch.rand(128, 64, 32, 32)
    m_q.input_scale.data, _ = reset_scale_and_zero_point(real_input, act_N_bits, 'unsigned')
    m_q.method = 'normal'
    real_output = m_q(real_input)
    real_gard = torch.autograd.grad(real_output.sum(), m_q.weight)[0]
    m_q.method = method
    quntized_output = m_q(real_input)
    quntized_gard = torch.autograd.grad(quntized_output.sum(), m_q.weight)[0]

    print("=================real_output -vs- int_output=================")
    print(f'    MSE: {torch.nn.functional.mse_loss(real_output, quntized_output)} ')
    print("==============grad(real_output), grad(int_output)==============")
    print(f'    Abs-Mean: {(quntized_gard-real_gard).abs().mean()} ')
    #assert (quntized_gard-real_gard).abs().mean() < 5e-2
    #print('* Test passed.')


def plot_real_dequantized_histogram(x,N_bits=4, log=True):
    methods = ["sym","SAWB","heuristic"]
    sns.set()
    fig, axes = plt.subplots(1, len(methods), figsize=(len(methods) * 5, 3))
    for j in range(3):
        method = methods[j]
        scale, zero_point = reset_scale_and_zero_point(x, N_bits, method)
        x_int = linear_quantize(x, scale, zero_point, N_bits)
        x_hat = linear_dequantize(x_int, scale, zero_point)
        
        unique_weight = np.unique(x_hat)
        density = np.zeros_like(unique_weight)
        for i in range(len(density)):
            density[i] = np.isclose(x_hat, np.ones_like(x_hat) * unique_weight[i]).sum()
        density = density / x_hat.numel()

        width = (max(unique_weight) - min(unique_weight)) / (2 * len(unique_weight))
        num_bins = max(16, len(unique_weight))
        counts, bins = np.histogram(np.array(x).reshape(-1), num_bins)
        width_ = (max(bins) - min(bins)) / num_bins
        counts = counts / x.numel()
        axes[j].bar(bins[:-1], counts, width=width_, alpha=0.4, label='real')
        if width < 0.03:
            width = width_
        axes[j].bar(unique_weight, density, width=width, label='quantized')
        if log:
            axes[j].set_yscale('log')
        mse = torch.nn.functional.mse_loss(x, x_hat).item()
        axes[j].set_title(method + '\n MSE:' + str(np.round(mse, 5)))
        axes[j].legend()
        _, xtick_idx = np.histogram(np.arange(len(unique_weight)), 5)
        axes[j].set_xticks(np.round(unique_weight, 2)[xtick_idx.astype(np.int)])

def plot_layers_histogram(quantized_model: torch.nn.Module, log: bool = True):
    module_names = ['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'fc']
    sns.set()
    fig, axes = plt.subplots(1, len(module_names), figsize=(len(module_names) * 5, 3))
    j = 0
    for name, m in quantized_model.named_modules():
        if name in module_names:
            m.weight_scale.data, zero_point = reset_scale_and_zero_point(m.weight, m.weight_N_bits, m.method)
            quantized_weight = linear_quantize(m.weight, m.weight_scale, zero_point.to(m.weight.device), m.weight_N_bits,True)
            weight_reconstructed = linear_dequantize(quantized_weight, m.weight_scale, zero_point)
            x = m.weight.detach().cpu()
            x_hat = weight_reconstructed.detach().cpu()

            unique_weight = np.unique(x_hat)
            density = np.zeros_like(unique_weight)
            for i in range(len(density)):
                density[i] = np.isclose(x_hat, np.ones_like(x_hat) * unique_weight[i]).sum()
            density = density / x_hat.numel()

            width = (max(unique_weight) - min(unique_weight)) / (2 * len(unique_weight))
            num_bins = max(16, len(unique_weight))
            counts, bins = np.histogram(np.array(x).reshape(-1), num_bins)
            width_ = (max(bins) - min(bins)) / num_bins
            counts = counts / x.numel()
            axes[j].bar(bins[:-1], counts, width=width_, alpha=0.4, label='real')
            if width < 0.03:
                width = width_
            axes[j].bar(unique_weight, density, width=width, label='quantized')
            if log:
                axes[j].set_yscale('log')
            mse = torch.nn.functional.mse_loss(x, x_hat).item()
            axes[j].set_title(name + '\n MSE:' + str(np.round(mse, 5)))
            axes[j].legend()
            _, xtick_idx = np.histogram(np.arange(len(unique_weight)), 5)
            axes[j].set_xticks(np.round(unique_weight, 2)[xtick_idx.astype(np.int)])
            j += 1
            

def input_activation_hook(model, data):
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (Quantized_Conv2d, Quantized_Linear)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(data)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def model_to_quant(model, calibration_loader, act_N_bits=8, weight_N_bits=8, method='sym',
                   device=torch.device("cuda"), bitwidth_dict: Dict = None):
    quantized_model = copy.deepcopy(model)
    input_activation, output_activation = input_activation_hook(quantized_model,
                                                                next(iter(calibration_loader))[0].to(device))
    for name, m in quantized_model.named_modules():
        if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
            if name != 'conv1':
                if bitwidth_dict is None:
                    m.weight_N_bits = weight_N_bits
                else:
                    m.weight_N_bits = bitwidth_dict[name]
                m.act_N_bits = act_N_bits
                m.method = method
                m.input_scale.data, _= reset_scale_and_zero_point(input_activation[name], m.act_N_bits, 'unsigned')
    return quantized_model
