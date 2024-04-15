## CNN implementation for first sprint 

import torch
import requests
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()


# Multiple examples to play around with
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# url, filename = ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Pomeranian.JPG/640px-Pomeranian.jpg", "doggo.jpg")
# url, filename = ("https://mein.toubiz.de/api/v1/article/f8024aa3-1228-4039-b3c7-9c5c91db86a1/mainImage?format=image/jpeg&width=1900", "hohenstaufen.jpg")
# url, filename = ("https://variety.com/wp-content/uploads/2021/04/Godzilla-2.jpg?w=1000&h=563&crop=1", "godzilla.jpg")
# url, filename = ("https://i.redd.it/a3f8536mbhga1.jpg", "spino.jpg")
# url, filename = ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Flag_of_Barbados.svg/1200px-Flag_of_Barbados.svg.png", "barbados.jpg")
# url, filename = ("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfFhunNBlqAWngaW_Ciucth3hbMg5m4tKKCFNTGek_xg&s", "wednesday.jpg")
# url, filename = ("https://img.welt.de/img/iconist/maenner/mobile184375916/9902504597-ci102l-w1024/Welt-Portrait-Shooting-2016.jpg", "poschardt.jpg")
# url, filename = ("https://www.apple.com/leadership/images/bio/tim-cook_image.png.large_2x.png", "tim_cook.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# Download ImageNet labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(url)

if response.status_code == 200:
    with open("imagenet_classes.txt", "wb") as f:
        f.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download file.")

with open("imagenet_classes.txt", "wb") as f:
    f.write(response.content)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())