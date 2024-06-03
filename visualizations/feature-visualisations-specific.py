import torch
import matplotlib.pyplot as plt
import os
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import render, objectives
import torch
from lucent.optvis import render, objectives
import matplotlib.pyplot as plt
import os
import torch.nn.utils.prune as prune
from lucent.modelzoo import util
import random
import torch.nn.utils.prune as prune


# Load default model and check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')

# Pruning amounts to compare
pruning_rates = [0.3, 0.5, 0.7]

def localUnstructuredL1Pruning(pruning_rate, model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return model

# Dictionary to hold pruned models
models = {}

# Add the GoogleNet to the models dictionary
models[1] = model

# L1 unstructured local pruning
for idx, amt in enumerate(pruning_rates, start = 1):  
    model_idx = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained = True)
    model_idx_pruned = localUnstructuredL1Pruning(amt, model_idx)
    models[idx + 1] = model_idx_pruned  

def get_googlenet_layers(model):
    # Create a dictionary to hold selected layer names and a randomly chosen filter index
    selected_layer_info = {}

    # Function to get information about inception modules
    def get_layer_info(module, prefix=''):
        for name, layer in module.named_children():
            layer_name = prefix + name
            if 'inception' in layer_name:
                # Check for convolutional layers within each inception module
                conv_layers = [l for n, l in layer.named_children() if isinstance(l, torch.nn.Conv2d)]
                if conv_layers:
                    # Randomly select one convolutional layer and one filter index
                    selected_layer = random.choice(conv_layers)
                    filter_index = random.randint(0, selected_layer.out_channels - 1)
                    # Format the layer name to exclude '2d'
                    conv_layer_name = selected_layer.__class__.__name__.replace('Conv2d', 'conv')
                    layer_name_underscore = layer_name.replace('.', '_') + '_' + conv_layer_name
                    selected_layer_info[layer_name_underscore] = filter_index

            # Recursively get info for submodules
            get_layer_info(layer, layer_name + '.')
    
    # Get info for the main model
    get_layer_info(model)
    
    return selected_layer_info

layers = get_googlenet_layers(model)

# print(layers)

# Loop through all the models and evaluate
for model_index, model in models.items():
    model_name = f"model_{model_index}"  # Define a descriptive model name based on the index
    model.to(device).eval()  # Set model to evaluation mode
    
    # Create output directory for each model
    output_dir = f"visualisations/{model_name}"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    print(f"Processing model: {model_name}")

    # Iterate over each layer and its associated filter index in the layers dictionary
    for layer_name, filter_index in layers.items():
        print(f"Evaluating layer: {layer_name} with selected filter index {filter_index}")

        # Set visualization goal for the specific layer and filter index
        obj = objectives.channel(layer_name, filter_index)

        try:
            # Render the image and save it
            image_list = render.render_vis(model, obj, show_inline=False, thresholds=(512,))  # Increase iterations if needed

            # Check if the image list is not empty and retrieve the first image
            if image_list:
                image = image_list[0][0]  # Access the first image in the returned list

                # Remove extra dimensions if any and prepare for saving
                image = image.squeeze()

                # Construct the filename and the path to save the image
                image_filename = f"{layer_name}_filter_{filter_index}_visualization.jpg"
                image_path = os.path.join(output_dir, image_filename)
                
                # Save the image
                plt.imshow(image)
                plt.axis('off')  # Hide axis for better visualization
                plt.savefig(image_path)  # Save image file
                plt.close()  # Close the plot to free resources

                print(f"Image saved at: {os.path.abspath(image_path)}")
            else:
                print(f"No image generated for filter {filter_index} in layer: {layer_name}")
        except Exception as e:
            print(f"Error visualizing filter {filter_index} in layer {layer_name}: {e}")