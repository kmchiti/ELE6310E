import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import copy
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict
import os


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                      ! DO NOT MODIFY THESE FUNCTIONS !
def conv_layer_generator(base_path: str = 'layer_prob_base.yaml', in_channels: int = 3, out_channels: int = 16,
                         kernel_size: int = 3, stride: int = 1, Height: int = 32, Width: int = 32,
                         save_path: str = 'conv1'):
    """
    Generate the yaml file for the conv layer
    Args:
        base_path: the path of the base yaml file
        in_channels: the number of input channels
        out_channels: the number of output channels
        kernel_size: the size of the kernel
        stride: the stride of the conv layer
        Height: the height of the input
        Width: the width of the input
        save_path: the path to save the generated yaml file

    """
    with open(base_path, 'r') as file:
        txt = file.read()
        txt_new = txt.replace('$in_channels$', str(in_channels))
        txt_new = txt_new.replace('$out_channels$', str(out_channels))
        txt_new = txt_new.replace('$kernel_size$', str(kernel_size))
        txt_new = txt_new.replace('$stride$', str(stride))
        txt_new = txt_new.replace('$Height$', str(Height))
        txt_new = txt_new.replace('$Width$', str(Width))
        with open(save_path+'.yaml', 'w') as output:
            output.write(txt_new)


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
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(data)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def Extract_Stats(path="timeloop-mapper.stats.txt"):
    """
    Extract the stats from timeloop-mapper.stats.txt
    Args:
        path: path to the stats file

    Returns: a dictionary of stats

    """
    mylines = []
    with open(path, 'rt') as myfile:
        for myline in myfile:
            if myline.find("Energy:") != -1:
                energy_total = float(myline[myline.find(":") + 2:myline.find("uJ")])
            elif myline.find("Cycles:") != -1:
                Cycles = int(myline[myline.find(":") + 2:])
            elif myline.find("EDP(J*cycle):") != -1:
                EDAP = float(myline[myline.find(":") + 2:])
            elif myline.find("GFLOPs (@1GHz):") != -1:
                GFLOPs = float(myline[myline.find(":") + 2:])

    return energy_total, Cycles, EDAP, GFLOPs

def Run_Accelergy(path_to_eyeriss_files='Q3'):
    current_path = os.getcwd()
    path_to_eyeriss_files = os.path.join(current_path, path_to_eyeriss_files)
    name_layers = os.listdir(os.path.join(path_to_eyeriss_files, 'prob'))
    os.system(f"rm -rf {current_path}/timeloop-model.stats.txt")
    energy_total = 0
    for l in name_layers:
        command = f"timeloop-mapper {path_to_eyeriss_files}/prob/{l} {path_to_eyeriss_files}/arch/components/*.yaml  " \
                  f"{path_to_eyeriss_files}/arch/eyeriss_like.yaml {path_to_eyeriss_files}/constraints/*.yaml  {path_to_eyeriss_files}/mapper/mapper.yaml" + " >/dev/null 2>&1"
        os.system(command)
        energy = Extract_Stats(path=f"{current_path}/timeloop-mapper.stats.txt")[0]
        print(f'energy consumption for {l[:-5]} : {energy} uJ')
        energy_total += energy
        os.system(f"rm -rf {current_path}/timeloop-model.stats.txt")
    print(f"Total energy consumption for ResNet-20: {energy_total} uJ")
    return energy_total
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def model_to_spars(model, prune_ratio_dict: Dict):
    """
    Prune the model according to the prune_ratio_dict
    Args:
        model:
        prune_ratio_dict:

    Returns:

    """
    sparsed_model = copy.deepcopy(model)
    for name, m in sparsed_model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name in prune_ratio_dict.keys():
                # prune the conv layer using `torch.nn.utils.prune.ln_structured` function.
                # Prune the output channels based on L2 norm of the weight.
                ##### WRITE CODE HERE #####
                pass
            else:
                warnings.warn(f"not found ratio for module {name}")
    return sparsed_model


def generate_resnet_layers(model, base_path='common/layer_prob_base.yaml',  path='Q3/prob'):
    """
    Generate the yaml file for the conv layer
    Args:
        path: target path
        model: the model
        base_path: the path of the base yaml file

    Returns:

    """
    data = torch.rand(1, 3, 32, 32).to(model.conv1.weight.device)
    input_activation, output_activation = input_activation_hook(model, data)

    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        os.system(f"rm -rf {path}/*")

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, 'weight_mask'):
                # compute number of zero channels
                ##### WRITE CODE HERE #####
                num_zero_channels = 0
            else:
                num_zero_channels = 0

            # generate the yaml file for the conv layer using the function `conv_layer_generator`
            ##### WRITE CODE HERE #####
