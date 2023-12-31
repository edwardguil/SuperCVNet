import argparse, torch, os
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
from models import CVNetRerank, GlobalNetwork
from datasets import GoogleLandMarks, PairedDataset
from losses import ClassificationLoss, MomentumContrastiveLoss
import torchvision.transforms as transforms

def train_rerank(model, backbone, dataset, num_classes, num_epochs, batch_size, learning_rate, device, **kwargs):
    save = not kwargs.get('no_weights', False) 
    progress = not kwargs.get('no_progress', False)
    
    paired_dataset = PairedDataset(dataset)
    paired_trainloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    cross_entropy_loss.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        avg_loss = 0
        for i, data in enumerate(paired_trainloader):
            inputs, positive_inputs, labels = data
            inputs, positive_inputs, labels = inputs.to(device), positive_inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                query_features = backbone(inputs, ret_intermediate=True)
                key_features = backbone(inputs, ret_intermediate=True)

            rank = model(query_features, key_features)
            
            print(rank)
            print(labels)
            loss = cross_entropy_loss(rank, labels)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if progress:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}', end='\r')
        
        # Print info
        if progress:
            print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {avg_loss/(i + 1):.4f}', end='\r')
            print('')

        if save:
            if not os.path.exists('./weights'):
                os.makedirs('./weights')
            torch.save(model.state_dict(), os.path.join("./weights", f'{model.__class__.__name__}{dataset.__class__.__name__}-{epoch}.pkl'))

    print('Finished Training')


def get_args():
    parser = argparse.ArgumentParser(description='Training script for rerank model')

    parser.add_argument('--model', type=str, default='CVNet', choices=['CVNet'], help='Name of the model to train')
    parser.add_argument('--backbone', type=str, default='CVNet', choices=['CVNet'], help='Name of the backbone for feature extraction')
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['GoogleLandmarks', 'Cifar10'], help='Name of the dataset to train on')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=144, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0015, help='Learning rate')
    parser.add_argument('--reduction_dim', type=int, default=2048, help='The reduction dim on the backbone')
    parser.add_argument('--resnet_depth', type=int, default=50, help='The depth of the underlying ResNet model')
    parser.add_argument('--no_save', action='store_true', default=False, help="Don't save the weights during training")
    parser.add_argument('--no_progress', action='store_true', default=False, help="Don't to print training progress")

    return parser.parse_args()


def launch_script(args):
    if args.model == 'CVNet':
        model = CVNetRerank()

    if args.backbone == 'CVNet':
        backbone = GlobalNetwork(reduction_dim=args.reduction_dim, resnet_depth=args.resnet_depth)  # you might need to customize the initialization

    transform = transforms.Compose([
        transforms.Resize(512),  # Resize the images to 512x512 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    if args.dataset == 'Cifar10':
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'GoogleLandmarks':
        dataset = GoogleLandMarks(root='./data', split='train', remove_tars=False, verify=False)
        num_classes = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_rerank(model=model, 
                backbone=backbone,
                dataset=dataset, 
                num_epochs=args.num_epochs, 
                num_classes=num_classes,
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate, 
                device=device,
                save=args.no_save,
                progress=args.no_progress
    )


if __name__ == '__main__':
    args = get_args()
    launch_script(args)