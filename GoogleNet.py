## implementation GoogleNet

# all required imports
import torch
import urllib
from PIL import Image
from torchvision import transforms

# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model.eval()

# URL for the imagenet_classes.txt file
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
filename = "imagenet_classes.txt"

# Download the imagenet_classes.txt file
urllib.request.urlretrieve(url, filename)

# Load and preprocess the example image
url_image = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
filename_image = "dog.jpg"
urllib.request.urlretrieve(url_image, filename_image)
input_image = Image.open(filename_image)
preprocess = transforms.Compose([
    # Resize the images to 256x256 pixels
    transforms.Resize(256),      
    # Crop the center of the images to obtain the required size of 224x224 pixels        
    transforms.CenterCrop(224),
    # Convert the images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize the tensors using the specified mean and standard deviation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Execute model without gradient calculations
with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# Read the categories from the imagenet_classes.txt file
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


     