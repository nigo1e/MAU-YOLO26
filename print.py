import torch
import torch.nn as nn
from ultralytics import YOLO
from tabulate import tabulate


def print_model_structure(model, prefix="", show_all=False):
    lines = []
    total_params = 0

    for name, module in model.named_children():
        full_name = f"{prefix}{name}"
        module_type = module.__class__.__name__
        param_count = sum(p.numel() for p in module.parameters())
        total_params += param_count
        info_parts = []

        if isinstance(module, nn.Conv2d):
            info_parts.append(f"in={module.in_channels}, out={module.out_channels}")
            info_parts.append(f"k={module.kernel_size}")
            if module.stride != (1, 1):
                info_parts.append(f"s={module.stride}")
            if module.padding != (0, 0):
                info_parts.append(f"p={module.padding}")
            if module.groups > 1:
                info_parts.append(f"groups={module.groups}")

        elif isinstance(module, nn.BatchNorm2d):
            info_parts.append(f"features={module.num_features}")
            if module.eps != 1e-5:
                info_parts.append(f"eps={module.eps}")
            if module.momentum != 0.1:
                info_parts.append(f"momentum={module.momentum}")

        elif isinstance(module, nn.Linear):
            info_parts.append(f"in={module.in_features}, out={module.out_features}")

        elif isinstance(module, nn.ReLU):
            info_parts.append("inplace" if module.inplace else "")
        elif isinstance(module, nn.LeakyReLU):
            info_parts.append(f"neg_slope={module.negative_slope}")

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            info_parts.append(f"k={module.kernel_size}")
            if module.stride != module.kernel_size:
                info_parts.append(f"s={module.stride}")
            if module.padding != 0:
                info_parts.append(f"p={module.padding}")

        elif isinstance(module, nn.Upsample):
            info_parts.append(f"scale={module.scale_factor}")
            info_parts.append(f"mode={module.mode}")

        info_str = ", ".join([p for p in info_parts if p])

        if show_all or param_count > 0 or info_str or isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            lines.append({
                "Layer": full_name,
                "Type": module_type,
                "Params": f"{param_count:,}",
                "Info": info_str
            })

        if list(module.children()):
            sub_lines, sub_params = print_model_structure(module, f"{full_name}.", show_all)
            lines.extend(sub_lines)
            total_params += sub_params

    return lines, total_params


def print_detailed_structure(model):
    print("=" * 120)
    print("YOLO Detailed Model Structure")
    print("=" * 120)

    lines, total_params = print_model_structure(model, show_all=False)

    table_headers = ["Layer", "Type", "Params", "Info"]
    table_data = []

    for line in lines:
        table_data.append([
            line["Layer"],
            line["Type"],
            line["Params"],
            line["Info"]
        ])

    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    print("-" * 120)
    print(f"Total Parameters: {total_params:,}")
    print("=" * 120)

    print("\nParameter Statistics by Module Type:")
    print("-" * 60)

    param_by_type = {}
    for line in lines:
        module_type = line["Type"]
        params = int(line["Params"].replace(",", "")) if line["Params"] != "0" else 0

        if module_type in param_by_type:
            param_by_type[module_type] += params
        else:
            param_by_type[module_type] = params

    sorted_types = sorted(param_by_type.items(), key=lambda x: x[1], reverse=True)

    for module_type, params in sorted_types:
        if params > 0:
            percentage = (params / total_params) * 100
            print(f"{module_type:20} {params:12,} ({percentage:5.1f}%)")


def print_compact_structure(model):
    print("=" * 100)
    print("YOLO Model Structure (Layers with Parameters)")
    print("=" * 100)

    lines, total_params = print_model_structure(model, show_all=False)
    param_lines = [line for line in lines if int(line["Params"].replace(",", "")) > 0]

    table_headers = ["Layer", "Type", "Params", "Info"]
    table_data = []

    for line in param_lines:
        table_data.append([
            line["Layer"],
            line["Type"],
            line["Params"],
            line["Info"]
        ])

    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    print("-" * 100)
    print(f"Total Parameters: {total_params:,}")

    conv_params = 0
    bn_params = 0
    linear_params = 0
    other_params = 0

    for line in param_lines:
        params = int(line["Params"].replace(",", ""))
        module_type = line["Type"]

        if "Conv" in module_type:
            conv_params += params
        elif "BatchNorm" in module_type:
            bn_params += params
        elif "Linear" in module_type:
            linear_params += params
        else:
            other_params += params

    print("\nParameter Distribution:")
    print(f"  Convolutional Layers: {conv_params:,} ({conv_params / total_params * 100:.1f}%)")
    print(f"  BatchNorm Layers:    {bn_params:,} ({bn_params / total_params * 100:.1f}%)")
    print(f"  Linear Layers:       {linear_params:,} ({linear_params / total_params * 100:.1f}%)")
    print(f"  Other Layers:        {other_params:,} ({other_params / total_params * 100:.1f}%)")


model = YOLO("ultralytics/cfg/models/26/yolo261111111.yaml")
target_model = model.model if hasattr(model, 'model') else model

print("\n" + "=" * 120)
print(f"Model: {target_model.__class__.__name__}")
print("=" * 120)

print_detailed_structure(target_model)
print_compact_structure(target_model)

print("\n" + "=" * 60)
print("Model Summary")
print("=" * 60)

trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in target_model.parameters())

print(f"Total Parameters:       {total_params:,}")
print(f"Trainable Parameters:   {trainable_params:,}")
print(f"Non-trainable:          {total_params - trainable_params:,}")
print(f"Trainable Percentage:   {trainable_params / total_params * 100:.2f}%")
print("=" * 60)