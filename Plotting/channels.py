import torchvision.models as models
import torch
# Laden des vortrainierten GoogleNet-Modells
googlenet = models.googlenet(pretrained=True)

# Untersuchen der Architektur des GoogleNet-Modells
for name, layer in googlenet.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        print(f"Layer: {name}, Out Channels: {layer.out_channels}")
