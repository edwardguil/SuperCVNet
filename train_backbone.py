import argparse
from helpers import train_backbone
from models import SuperCVNetGlobal, CVNetGlobal
from datasets import GoogleLandMarks
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Training script for model backbones')

    parser.add_argument('--model', type=str, default='CVNetGlobal', help='Name of the model to train')
    parser.add_argument('--dataset', type=str, default='Cifar10', help='Name of the dataset to train on')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=144, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0015, help='Learning rate')
    parser.add_argument('--tau', type=float, default=1/30, help='Tau for losses')
    parser.add_argument('--lambda_cls', type=float, default=0.5, help='Lambda for classification loss')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='Lambda for contrastive loss')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momemtum for contrastive loss')
    parser.add_argument('--reduction_dim', type=int, default=2048, help='Momemtum for contrastive loss')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Here you can add code to initialize your model and dataset based on the input arguments
    if args.model.lower() == 'SuperCVNetGlobal'.lower():
        model = SuperCVNetGlobal(reduction_dim=args.reduction_dim, momentum=args.momentum)  # you might need to customize the initialization
    elif args.model.lower() == 'CVNetGlobal'.lower():
        model = CVNetGlobal(reduction_dim=args.reduction_dim, momentum=args.momemtum)
    else:
        raise ValueError(f"Model {args.model} dosen't exist. Only: 'SuperCVNetGlobal' or 'CVNetGlobal'")

    transform = transforms.Compose([
        transforms.Resize(512),  # Resize the images to 512x512 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    if args.dataset.lower() == 'Cifar10'.lower():
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset.lower() == 'GoogleLandmarks'.lower():
        dataset = GoogleLandMarks(root='./data', split='train', remove_tars=False, verify=False)
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {args.dataset} dosen't exist. Only: 'Cifar10' or 'GoogleLandmarks'")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_backbone(model=model, 
                dataset=dataset, 
                num_epochs=args.num_epochs, 
                num_classes=num_classes,
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate, 
                device=device,
                tau=args.tau, 
                lambda_cls=args.lambda_cls, 
                lambda_con=args.lambda_con)
