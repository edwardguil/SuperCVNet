import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import CVNetGlobal
from losses import ClassificationLoss, MomentumContrastiveLoss
from datasets import DatasetPaired
import sys

def main(lr):
    # Hyperparameters
    tau = 1/30
    momentum = 0.999
    lambda_cls = 0.5
    lambda_con = 0.5
    learning_rate = 0.0015 # 0.005625 paper learning rate
    batch_size = 2 # 144 paper batch size
    num_epochs = 25

    reduction_dim = 2048
    num_classes = 10

    if lr:
        learning_rate = lr
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(64),  # Resize the images to 256x256 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    # Initialize model
    model = CVNetGlobal(reduction_dim)

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    paired_train = DatasetPaired(train)
    paired_trainloader = DataLoader(paired_train, batch_size=batch_size, shuffle=True)

    # Moving model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    classification_loss_fn = ClassificationLoss(num_classes, tau=tau)
    momentum_contrastive_loss_fn = MomentumContrastiveLoss(num_classes, tau=tau)

    # Losses contain parameters that need to be sent to device
    classification_loss_fn.to(device)
    momentum_contrastive_loss_fn.to(device)

    optimizer = optim.SGD(model.global_network.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        avg_cls_loss, avg_con_loss = 0, 0
        for i, data in enumerate(paired_trainloader):
            # Sample data and sent to device
            inputs, positive_inputs, labels = data
            inputs, positive_inputs, labels = inputs.to(device), positive_inputs.to(device), labels.to(device)

            # Zero optimizer gradient for new 
            optimizer.zero_grad()

            # Forward pass
            global_features, momentum_features = model(inputs, positive_inputs)

            # Calculate individual losses
            loss_cls = classification_loss_fn(global_features, labels)
            loss_con = momentum_contrastive_loss_fn(global_features, momentum_features, labels)

            # Total loss
            loss = lambda_cls * loss_cls + lambda_con * loss_con

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            momentum_dict = {k: v.data for k, v in model.global_network.state_dict().items()}
            model.momentum_network.load_state_dict(momentum_dict)

            # Save info
            avg_cls_loss += loss_cls.item()
            avg_con_loss += loss_con.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {loss_cls:.4f}, Contrast Loss: {loss_con:.4f}', end='\r')
        
        # Print info
        print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {avg_cls_loss/(i + 1):.4f}, Contrast Loss: {avg_con_loss/(i + 1):.4f}', end='\r')
        print('')
        # As per paper, queue is reset on each iteration
        momentum_contrastive_loss_fn.clear_queue()
        momentum_contrastive_loss_fn.to(device)

        # Save weights
        torch.save(model.state_dict(), f'CVNetBackboneCifar10-{epoch}.pkl')

    print('Finished Training')


if __name__ == "__main__":
    lr = None
    if "lr" in sys.argv[0]:
        lr = float(sys.argv[0][2:])

    main(lr)

