#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
#%%
# Set the seed for reproducibility
seed = 33
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
# Define the CNN model
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
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Initialize the model
model = LeNet5().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
best_accuracy = 0.0
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculate accuracy on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    # Save the model if it has the best accuracy on the validation set
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # torch.save(model.state_dict(), 'best_model.pth')
        save_path = f'models/best_model_seed{seed}.pth'
        torch.save(model.state_dict(), f'/u/scratch/p/pterway/UCLAProjects/Slivit/CKA-Centered-Kernel-Alignment/models/best_model_seed{seed}.pth')
    
    print(f"Epoch {epoch+1}: Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%")

print('Training finished.')
#%%
# Loading the saved model
loaded_model = LeNet5().to(device)
loaded_model.load_state_dict(torch.load(f'/u/scratch/p/pterway/UCLAProjects/Slivit/CKA-Centered-Kernel-Alignment/models/best_model_seed{seed}.pth'))
loaded_model.eval()
#
#%%
# evaluate performance on the test data now
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")
# %%
