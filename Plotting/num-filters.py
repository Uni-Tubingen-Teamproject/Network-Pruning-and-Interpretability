import torch
import torch.nn as nn

# Beispielmodell: GoogLeNet (Inception v1)
from torchvision.models import googlenet

# Modell laden
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                       weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)

# Funktion zum Ausgeben der Dimensionen der Konvolutionsgewichte


def print_conv_dimensions(model):
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            out_channels, in_channels, height, width = weight.size()
            print(f"{module_name}: ({out_channels}, {
                  height}, {width}, {in_channels})")


# Dimensionen der Konvolutionsgewichte ausgeben
print_conv_dimensions(model)
