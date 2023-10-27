import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import CVNetGlobal
from losses import ClassificationLoss, MomentumContrastiveLoss
from datasets import PairedCIFAR10
from helpers import load_weights

def main():
    # Hyperparameters
    tau = 1/30
    batch_size = 16 # 144 paper batch size
    num_epochs = 1

    reduction_dim = 2048
    num_classes = 10

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(64),  # Resize the images to 256x256 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49421427, 0.48513183, 0.45040932), (0.24665256, 0.24289224, 0.26159248))
    ])

    # Initialize model
    model = CVNetGlobal(reduction_dim)
    load_weights(model, './weights/CVNetBackboneCifar10-24.pkl')



    train = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    paired_train = PairedCIFAR10(train, file="./data/cifar10_pair_cache_test.json")
    paired_trainloader = DataLoader(paired_train, batch_size=batch_size, shuffle=True)

    # Moving model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    classification_loss_fn = ClassificationLoss(num_classes, tau=tau)
    momentum_contrastive_loss_fn = MomentumContrastiveLoss(num_classes, tau=tau)

    # Losses contain parameters that need to be sent to device
    classification_loss_fn.to(device)
    momentum_contrastive_loss_fn.to(device)

    # Training loop
    with torch.no_grad():
        for epoch in range(num_epochs):
            avg_cls_loss, avg_con_loss = 0, 0
            for i, data in enumerate(paired_trainloader):
                # Sample data and sent to device
                inputs, positive_inputs, labels = data
                inputs, positive_inputs, labels = inputs.to(device), positive_inputs.to(device), labels.to(device)

                # Forward pass
                global_features, momentum_features = model(inputs, positive_inputs)

                # Calculate individual losses
                loss_cls = classification_loss_fn(global_features, labels)
                loss_con = momentum_contrastive_loss_fn(global_features, momentum_features, labels)

                momentum_dict = {k: v.data for k, v in model.global_network.state_dict().items()}
                model.momentum_network.load_state_dict(momentum_dict)

                # Save info
                avg_cls_loss += loss_cls.item()
                avg_con_loss += loss_con.item()
                print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {loss_cls:.4f}, Contrast Loss: {loss_con:.4f}', end='\r')
            
            # Print info
            print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {avg_cls_loss/(i + 1):.4f}, Contrast Loss: {avg_con_loss/(i + 1):.4f}', end='\r')
            print('')

    print('Finished Testing')


if __name__ == "__main__":
    main()

