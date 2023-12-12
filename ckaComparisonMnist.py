#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
#%%
import cca_core
from CKA import linear_CKA, kernel_CKA
#%%
# from cnnTrainingMnist import LeNet5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1) # Input channels changed to 1 for grayscale MNIST
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # AvgPool replaced with MaxPool
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # Updated output size based on new input size
        self.dropout1 = nn.Dropout(0.25) # Added dropout layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.25) # Added dropout layer
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
model_1 = LeNet5()
model_path = 'models/best_model.pth'
model_1.load_state_dict(torch.load(model_path))

# Move the model to the specified device
model_1.to(device)
#%%
model_2 = LeNet5()
model_path = 'models/best_model_seed33.pth'
model_2.load_state_dict(torch.load(model_path))
# Move the model to the specified device
model_2.to(device)

#%%
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# %%
# make prediction on the test set and compare the accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model_2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   
print('Accuracy of the network on the 10000 test images: %d %%' % (100. * correct / total))
# %%
# define the hook function
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
# %%
layer_name_for_activation = ['relu1', 'relu2', 'relu3', 'relu4']
for layer in layer_name_for_activation:
    model_1._modules[layer].register_forward_hook(get_activation(layer))

# %%
def get_activation(model, images, layer_name):
    activation = {}

    def hook(model, input, output):
        activation[layer_name] = output.detach()

    layer = model._modules[layer_name]
    layer.register_forward_hook(hook)

    model(images)

    return activation
#%%
layer_name_for_activation = ['relu1', 'relu2', 'relu3', 'relu4']

# %%
heatmap_comparisn_similarity = np.zeros((len(layer_name_for_activation), len(layer_name_for_activation)))
from collections import defaultdict
cka_score = defaultdict(list)
# %%

for i, layer1 in enumerate(layer_name_for_activation):
    for j, layer2 in enumerate(layer_name_for_activation):
        print(layer1, layer2)
        activation_model1 = get_activation(model_1, images, layer1)[layer1]
        activation_model2 = get_activation(model_2, images, layer2)[layer2]
        activation_model1_flatten = activation_model1.view(activation_model1.shape[0], -1)
        activation_model1_flatten_np = activation_model1_flatten.cpu().numpy()
        activation_model2_flatten = activation_model2.view(activation_model2.shape[0], -1)
        activation_model2_flatten_np = activation_model2_flatten.cpu().numpy()
        # avg_acts1 = np.mean(activation_model1, axis=(1,2))
        # avg_acts2 = np.mean(activation_model2, axis=(1,2))
        cka_score[(layer1,layer2)].append(linear_CKA(activation_model1_flatten_np,
                                                      activation_model2_flatten_np))


# %%

activation_model1 = {}
activation_model2 = {}
for l in layer_name_for_activation:
    activation_model1[l] = get_activation(model_1, images, l)[l]
    activation_model2[l] = get_activation(model_2, images, l)[l]



# %%
layer_name_for_activation = ['relu1', 'relu2', 'relu3', 'relu4']
activations = {}
for layer_name in layer_name_for_activation:
        activations[layer_name] = get_activation(model_1, images, layer_name)[layer_name]
# %%    
output_model1 = model_1(images)
output_model2 = model_2(images)
print(activation['relu2'].shape)
# %%
