import torch
from lucent.optvis import render, param, transform, objectives
import matplotlib.pyplot as plt
import os
import torch.nn.utils.prune as prune

# load default model and check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                       weights='GoogLeNet_Weights.DEFAULT')

# fixed pruning amount, chosen arbitrarily


rate = 0.8
pruning_rate = "0.8"

def localUnstructuredL1Pruning(pruning_rate, model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=rate)
    return model


def localStructuredL1Pruning(pruning_rate, model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight',
                                amount=rate, n=1, dim=0)
    return model


# Load other models

optimizer = "SGD"
retraining_epochs = "50"
pruning_method = "Local_Structured"
model_path_2 = f'./Pruned_Models/{pruning_method}/{optimizer}/{retraining_epochs}_Epochs/pruned_{
    pruning_rate}_local_structured_{optimizer}_retrained_{retraining_epochs}_epochs_model.pth'
model_2 = torch.load(model_path_2, map_location=device)
model_2.eval()

pruning_method = "Connection_Sparsity"
model_path_3 = f'./Pruned_Models/{pruning_method}/{optimizer}/{retraining_epochs}_Epochs/pruned_{
    pruning_rate}_connection_sparsity_{optimizer}_retrained_{retraining_epochs}_epochs_model.pth'
model_3 = torch.load(model_path_3, map_location=device)
model_3.eval()


# L2 structured local pruning
model_structured = torch.hub.load('pytorch/vision:v0.10.0',
                         'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model_structured_pruned = localStructuredL1Pruning(rate, model_structured)

models = [ ('local_structured_pruned', model_structured_pruned), ('local_structured_retrained', model_2)]
#models = [('local_structured_pruned', model_structured_pruned),]
# ('default', model),
# gather the names and filter number of all conv layers in the GoogleNet


def get_conv_layers_info(model):
    conv_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            formatted_name = name.replace('.', '_')
            conv_layers.append((formatted_name, layer.out_channels))
    return conv_layers

# print(get_conv_layers_info(model))


# loop through all the models and evaluate
for model_name, model in models:
    # set model to evaluation mode
    model.to(device).eval()
    conv_layers_info = get_conv_layers_info(model)

    # create output directory for each model
    output_dir = f"visualisations/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # evaluate every conv layer
    for name, num_filters in conv_layers_info:
        if name != 'inception5a_branch4_1_conv':
            continue

        channel = 118

        # set visualisation goal
        obj = objectives.channel(name, channel)

        # render the image and save it in list
        image_list = render.render_vis(model, obj, show_inline=False)

        # indice the list to get the image
        image = image_list[0]

        # remove a dimension to show image in matplotlib
        image = image.squeeze()

        # choose an image filename including layer and channel
        image_filename = f"{name}channel{channel}_visualisation_{rate}.jpg"

        # define storage path
        image_path = os.path.join(output_dir, image_filename)

        # Show image
        plt.imshow(image)
        plt.axis('off')  # hide axis
        plt.savefig(image_path)  # save image
        plt.close()  # close plot to avoid memory leak

        print(f"Image saved at: {os.path.abspath(image_path)}")

print("Feature visualisations completed and saved.")
