import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import CVNetGlobal
from losses import ClassificationLoss, MomentumContrastiveLoss
from datasets import PairedCIFAR10

# Hyperparameters
tau = 1/30
momentum = 0.999
lambda_cls = 0.5
lambda_con = 0.5
learning_rate = 0.005625
batch_size = 32 
num_epochs = 25


num_classes = 10
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize the images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
paired_trainset = PairedCIFAR10(trainset)
paired_trainloader = DataLoader(paired_trainset, batch_size=batch_size, shuffle=True)


# Initialize your model
model = CVNetGlobal(num_classes)

# Moving model to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

classification_loss_fn = ClassificationLoss(num_classes, tau=tau)
momentum_contrastive_loss_fn = MomentumContrastiveLoss(num_classes, tau=tau) 

optimizer = optim.SGD(model.global_network.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(paired_trainloader, 0):
        inputs, positive_inputs, labels = data
        inputs, positive_inputs, labels = inputs.to(device), positive_inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        global_features, momentum_features = model(inputs, positive_inputs)
        print(torch.isnan(global_features).any())
        print(torch.isnan(momentum_features).any())

        # Calculate loss
        loss_cls = classification_loss_fn(global_features, labels)
        loss_con = momentum_contrastive_loss_fn(global_features, momentum_features, labels)

        # Total loss
        loss = lambda_cls * loss_cls + lambda_con * loss_con
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
        # Print statistics
        print(f"[{epoch + 1}, {i + 1}] class_loss: {loss_cls.item()} contrast: {loss_con.item()}")
    
    # As per paper, queue is reset on each iteration
    momentum_contrastive_loss_fn.clear_queue()

print('Finished Training')
