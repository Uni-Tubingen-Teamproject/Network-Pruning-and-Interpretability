import torch
from lucent.optvis import render, objectives
import matplotlib.pyplot as plt
import os
import torch.nn.utils.prune as prune
from lucent.modelzoo import util
import random

# Load default model and check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')

# Fixed pruning amount, chosen arbitrarily
pruning_rate = 0.5

def localUnstructuredL1Pruning(pruning_rate, model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return model

def localStructuredL2Pruning(pruning_rate, model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
    return model

# L1 unstructured local pruning
model_1 = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model_1_pruned = localUnstructuredL1Pruning(pruning_rate, model_1)

# L2 structured local pruning
model_2 = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model_2_pruned = localStructuredL2Pruning(pruning_rate, model_2)

models = [('default', model), ('l1_pruned', model_1_pruned), ('l2_pruned', model_2_pruned)]

def get_googlenet_layers(model):
    
    # Create a dictionary to hold layer names and their output channels
    layer_info = {}
    
    # Recursive function to get layer names and output channels
    def get_layer_info(module, prefix=''):
        for name, layer in module.named_children():
            layer_name = prefix + name
            if isinstance(layer, torch.nn.Conv2d):
                layer_name_underscore = layer_name.replace('.', '_')
                layer_info[layer_name_underscore] = layer.out_channels
                
            # Recursively get info for submodules
            get_layer_info(layer, layer_name + '.')
    
    # Get info for the main model
    get_layer_info(model)
    
    return layer_info

# Loop through all the models and evaluate
for model_name, model in models:
    # Set model to evaluation mode
    model.to(device).eval()
    conv_layers_info = get_googlenet_layers(model)
    
    # Create output directory for each model
    output_dir = f"visualisations/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing model: {model_name}")
    print(f"Conv layers info: {conv_layers_info}")

    # Evaluate every conv layer
    for layer_name, num_filters in conv_layers_info.items():
        print(f"Evaluating layer: {layer_name} with {num_filters} filters")
        for channel in range(num_filters):
            print(f"Visualising channel: {channel} in layer: {layer_name}")
            # Set visualization goal
            obj = objectives.channel(layer_name, channel)

            try:
                # Render the image and save it in list
                image_list = render.render_vis(model, obj, show_inline=False, thresholds=(512,))  # Increase iterations

                # Indice the list to get the image
                if image_list:
                    image = image_list[0]

                    # Remove a dimension to show image in matplotlib
                    image = image.squeeze()

                    # Choose an image filename including layer and channel
                    image_filename = f"{layer_name}_channel_{channel}_visualisation.jpg"
                    
                    # Define storage path
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # Show image
                    plt.imshow(image)
                    plt.axis('off')  # Hide axis
                    plt.savefig(image_path)  # Save image
                    plt.close()  # Close plot to avoid memory leak

                    print(f"Image saved at: {os.path.abspath(image_path)}")
                else:
                    print(f"No image generated for channel: {channel} in layer: {layer_name}")
            except Exception as e:
                print(f"Error visualising channel {channel} in layer {layer_name}: {e}")
