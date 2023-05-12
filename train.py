import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import Net
# Load the MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                          shuffle=False)



net = Net()

# Use CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train the model
for epoch in range(50):  # Loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()  
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  

print('Finished Training')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct / total}%')

torch.save(net,'mnist.pth')