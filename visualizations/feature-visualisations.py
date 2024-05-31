import torch
from lucent.optvis import render, param, transform, objectives
import matplotlib.pyplot as plt
import os
import torch.nn.utils.prune as prune

# load default model and check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')

# fixed pruning amount, chosen arbitrarily
pruning_rate = 0.4

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

# gather the names and filter number of all conv layers in the GoogleNet
def get_conv_layers_info(model):
    conv_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append((name, layer.out_channels))
    return conv_layers

# loop through all the models and evaluate
for model_name, model in models:
    # set model to evaluation mode
    model.to(device).eval()
    conv_layers_info = get_conv_layers_info(model)
    
    # create output directory for each model
    output_dir = f"visualizations/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # evaluate every conv layer
    for name, num_filters in conv_layers_info:
        for channel in range(num_filters):
            # set visualisation goal
            obj = objectives.channel(name, channel)

            # render the image and save it in list
            image_list = render.render_vis(model, obj, show_inline=False)

            # indice the list to get the image
            image = image_list[0]

            # remove a dimension to show image in matplotlib
            image = image.squeeze()

            # choose an image filename including layer and channel
            image_filename = f"{name}_channel_{channel}_visualisation.jpg"
            
            # define storage path
            image_path = os.path.join(output_dir, image_filename)
            
            # Show image
            plt.imshow(image)
            plt.axis('off')  # hide axis
            plt.savefig(image_path)  # save image
            plt.close()  # close plot to avoid memory leak

            print(f"Image saved at: {os.path.abspath(image_path)}")

print("Feature visualisations completed and saved.")
